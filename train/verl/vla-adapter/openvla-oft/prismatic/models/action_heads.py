"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.models.diffusion_transformer import  DiT_SingleTokenAction_OneCtx


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


class FlowMatchingActionHead(nn.Module):
    """
    MLP-based action head that generates continuous actions via flow matching.
    
    Flow matching is an alternative to diffusion models that directly learns the continuous-time flow
    between a simple distribution (e.g., standard normal) and the target distribution.
    
    Inspired by: https://arxiv.org/abs/2210.02747 and PI0: A Vision-Language-Action Flow Model
    """
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_flow_steps=10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_flow_steps = num_flow_steps
        
        self.flow_predictor = FlowPredictionDiT_V1(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=512, action_dim=action_dim
        )

        # Time encoder for positional encoding of timesteps
        self.time_encoder = nn.Identity()

    def sample_noise(self, shape, device):
        """Sample noise from a standard normal distribution."""
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )
        return noise
    

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.bfloat16, device=device)
        
    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the flow prediction network. Returns noise, noisy actions, and the
        corresponding flow timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        
        # Sample random noise with shape equal to actions
        noise = self.sample_noise((batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device)
        
        # Sample random flow timesteps (one for each action in batch)
        timesteps = self.sample_time(batch_size, device)
        
        # In flow matching, we interpolate between noise and ground truth based on timestep
        # x_t = (1-t) * noise + t * ground_truth
        time_expanded = timesteps.view(-1, 1, 1)
        noisy_actions = (1 - time_expanded) * noise + time_expanded * ground_truth_actions
        u_t = noise - ground_truth_actions

        timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        timestep_embeddings = timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)
        
        return_dict = dict(
            noise=noise,
            flow=u_t,
            noisy_actions=noisy_actions,
            timestep_embeddings=timestep_embeddings,
        )
        
        return return_dict
    
    def predict_flow(self, actions_hidden_states, noisy_actions=None, timestep_embeddings=None, 
                    noisy_action_projector=None, proprio=None, proprio_projector=None):
        """
        Given a batch of last hidden Transformer layer embeddings, predicts the flow field
        that transforms the noisy actions to the target actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        if noisy_actions is not None:
            noisy_actions_flat = noisy_actions.reshape(batch_size, -1).unsqueeze(-1).to(torch.bfloat16)  # (bsz, chunk_len * action_dim, 1)
            noise_actions_hidden_states = noisy_action_projector(noisy_actions_flat)  # (B, chunk_len * action_dim, llm_dim)
        else:
            noise_actions_hidden_states = torch.zeros_like(actions_hidden_states)
        
        if proprio is not None and proprio_projector is not None:
            proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
        else:
            proprio_features = None
        
        rearranged_actions_hidden_states = noise_actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        
        flow_pred = self.flow_predictor(
            obs=rearranged_actions_hidden_states,
            hidden_states=actions_hidden_states,
            time_step=timestep_embeddings,
            proprio_states=proprio_features
        )
        
        return flow_pred
    
    def sample_actions(self, actions_hidden_states, num_steps=10, noisy_action_projector=None, 
                      proprio=None, proprio_projector=None):
        """
        Samples actions by integrating the flow field from noise to the target distribution.
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # Start from standard normal noise
        x = self.sample_noise((batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device)
        
        # Discretize the flow into num_steps
        dt = -1.0 / num_steps  # Negative step size for reverse flow
        
        # Start from t=1 and move backward to t=0
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        # Euler integration of the flow field
        while time >= -dt / 2:
            # Current timestep in [0, 1]
            current_t = time.expand(batch_size)
            
            # Get timestep embeddings
            timestep_embeddings = self.time_encoder(current_t).to(device)
            timestep_embeddings = timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)
            
            # Predict flow at current position
            flow = self.predict_flow(
                actions_hidden_states, 
                noisy_actions=x, 
                timestep_embeddings=timestep_embeddings,
                noisy_action_projector=noisy_action_projector,
                proprio=proprio,
                proprio_projector=proprio_projector
            )
            
            # Update position using Euler step
            x = x + dt * flow
            time += dt
        
        return x
    

class FlowPredictionDiT_V1(nn.Module):
    """
    Diffusion flow prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a flow prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.dit = DiT_SingleTokenAction_OneCtx(in_channels=transformer_hidden_dim, out_channels=action_dim, depth=8, hidden_size=hidden_dim, num_heads=8, ctx_every=2)

    def forward(
        self,
        obs, hidden_states = None, time_step=None, proprio_states = None
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        # output: predicted noise

        output = self.dit(x = obs, context = hidden_states, timesteps = time_step, proprio = proprio_states)
        return output
