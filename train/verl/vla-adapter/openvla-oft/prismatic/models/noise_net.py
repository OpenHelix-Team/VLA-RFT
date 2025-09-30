# noise_net.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.vla.constants import (
    ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX,
    NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS
)

__all__ = ["TokenSigmaNet"]

from prismatic.models.diffusion_transformer import DiT_SingleTokenAction_OneCtx


class TokenSigmaDiT_V1(nn.Module):
    """
    DiT model for predicting log_std values for each action dimension.
    """
    def __init__(
        self,
        llm_hidden_dim: int,     # H_llm (e.g., 896)
        *,
        action_dim: int = ACTION_DIM,
        depth: int = 8,
        hidden_size: int = 512,
        num_heads: int = 8,
        ctx_every: int = 2,
    ):
        super().__init__()
        in_channels = action_dim * llm_hidden_dim
        self.dit = DiT_SingleTokenAction_OneCtx(
            in_channels=in_channels,
            out_channels=action_dim,   # output channels per dimension
            depth=depth,
            hidden_size=hidden_size,
            num_heads=num_heads,
            ctx_every=ctx_every,
        )

    def forward(
        self,
        obs,                    # (B, chunk_len, ACTION_DIM*H_llm)
        hidden_states=None,     # (B, L, T_ctx, H_llm) or (B, T_ctx, H_llm)
        time_step=None,         # shape according to your implementation
        proprio_states=None     # (B, 1, H_llm)
    ):
        return self.dit(x=obs, context=hidden_states, timesteps=time_step, proprio=proprio_states)  # (B, chunk_len, ACTION_DIM)


class TokenSigmaNet(nn.Module):
    """
    Predicts per-dimension standard deviation σ (diagonal Gaussian) for each action token,
    output shape (B, chunk_len, ACTION_DIM).
    Usage similar to predict_flow:
      std = sigma_net(all_hidden_states, noisy_actions, timestep_embeddings, noisy_action_projector, proprio, proprio_projector)
    """
    def __init__(
        self,
        *,
        llm_hidden_dim: int,          # Required: e.g., 896
        min_std: float = 1e-3,
        max_std: float = 5e-1,
        depth: int = 8,
        num_heads: int = 8,
        hidden_size: int = 512,
        ctx_every: int = 2,
        clamp_min: float = 1e-6,
    ):
        super().__init__()
        assert min_std > 0 and max_std >= min_std

        self.llm_hidden_dim = int(llm_hidden_dim)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.clamp_min = float(clamp_min)

        # Stable mapping range for log σ
        self.register_buffer("log_std_min", torch.tensor(math.log(self.min_std), dtype=torch.float32))
        self.register_buffer("log_std_max", torch.tensor(math.log(self.max_std), dtype=torch.float32))

        # Initialize DiT immediately (non-lazy initialization)
        self.std_predictor = TokenSigmaDiT_V1(
            llm_hidden_dim=self.llm_hidden_dim,
            action_dim=ACTION_DIM,
            depth=depth, hidden_size=hidden_size, num_heads=num_heads, ctx_every=ctx_every,
        )

    @staticmethod
    def _ensure_ctx_shape(actions_hidden_states: torch.Tensor) -> torch.Tensor:
        """Ensure shape is (B, L, T_ctx, H_llm), preserving your 4D structure (including visual patches + action tokens)."""
        if actions_hidden_states.dim() == 3:
            B, T, H = actions_hidden_states.shape
            return actions_hidden_states.view(B, 1, T, H)
        elif actions_hidden_states.dim() == 4:
            return actions_hidden_states
        else:
            raise AssertionError("actions_hidden_states should be (B, T, H) or (B, L, T, H)")

    def _prep_obs_from_noisy(
        self,
        noisy_actions: torch.Tensor,               # (B, chunk_len, ACTION_DIM)
        noisy_action_projector: nn.Module,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        noisy_actions -> flatten (B, chunk_len*ACTION_DIM, 1)
                      -> projector (B, chunk_len*ACTION_DIM, H_llm)
                      -> reshape   (B, chunk_len, ACTION_DIM*H_llm)
        """
        B, chunk_len, action_dim = noisy_actions.shape
        assert action_dim == ACTION_DIM, f"action_dim={action_dim} inconsistent with ACTION_DIM={ACTION_DIM}"

        T_total = chunk_len * action_dim
        noisy_flat = noisy_actions.reshape(B, T_total).unsqueeze(-1).to(dtype)   # (B, T_total, 1)
        hidden_noisy = noisy_action_projector(noisy_flat)                        # (B, T_total, H_llm)
        assert hidden_noisy.size(-1) == self.llm_hidden_dim, \
            f"projector output H={hidden_noisy.size(-1)} inconsistent with llm_hidden_dim={self.llm_hidden_dim}"

        obs = hidden_noisy.reshape(B, chunk_len, action_dim * self.llm_hidden_dim)
        return obs

    def predict_std(
        self,
        actions_hidden_states: torch.Tensor,          # (B, 1, T_ctx, H_llm) or (B, T_ctx, H_llm)
        noisy_actions: torch.Tensor,                  # (B, chunk_len, ACTION_DIM)
        timestep_embeddings: Optional[torch.Tensor] = None,
        noisy_action_projector: Optional[nn.Module] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Returns std: (B, chunk_len, ACTION_DIM)
        """
        assert noisy_action_projector is not None, "noisy_action_projector is required"

        # context: preserve your concatenated visual patch 4D shape
        ctx = self._ensure_ctx_shape(actions_hidden_states)          # (B, L, T_ctx, H_llm)
        B = ctx.size(0)
        orig_dtype = ctx.dtype

        # obs: construct from noisy_actions following predict_flow pattern
        obs = self._prep_obs_from_noisy(noisy_actions, noisy_action_projector, dtype=orig_dtype)  # (B, chunk_len, ACTION_DIM*H)

        # Optional proprio
        proprio_states = None
        if proprio is not None and proprio_projector is not None:
            proprio_states = proprio_projector(proprio.reshape(B, -1).to(orig_dtype)).unsqueeze(1)  # (B,1,H_llm)

        # Compute in float32, then cast result back
        obs_ = obs.to(torch.float32)
        ctx_ = ctx.to(torch.float32)
        t_   = None if timestep_embeddings is None else timestep_embeddings.to(torch.float32)
        prop_= None if proprio_states is None else proprio_states.to(torch.float32)

        # DiT outputs per-dimension log_std raw values: (B, chunk_len, ACTION_DIM)
        log_std_raw = self.std_predictor(
            obs=obs_, hidden_states=ctx_, time_step=t_, proprio_states=prop_
        )
        if log_std_raw.dim() != 3 or log_std_raw.size(-1) != ACTION_DIM:
            raise RuntimeError(f"DiT output shape mismatch: {tuple(log_std_raw.shape)}, expected (B, chunk_len, {ACTION_DIM})")

        # tanh → [log_std_min, log_std_max] → σ
        squashed = torch.tanh(log_std_raw)  # (-1, 1)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * (squashed + 1.0) * 0.5
        std = torch.exp(log_std)  # (B, chunk_len, ACTION_DIM)

        return std.to(orig_dtype), log_std.to(orig_dtype)

    # Compatible with module(...) calls
    def forward(self, *args, **kwargs):
        return self.predict_std(*args, **kwargs)
