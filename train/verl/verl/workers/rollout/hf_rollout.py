import contextlib
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.torch_functional import get_response_mask
from .base import BaseRollout

from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


__all__ = ['HFRollout']


def _unwrap(m: nn.Module) -> nn.Module:  # >>> NEW: Compatible with DDP/FSDP wrapping
    return m.module if hasattr(m, "module") else m


class HFRollout(BaseRollout):

    def __init__(self, module: nn.Module, config,
                 action_head: nn.Module, proprio_projector: nn.Module,
                 noisy_action_projector: nn.Module, sigma_net: nn.Module):
        super().__init__()
        self.config = config
        self.module = module
        self.action_head = _unwrap(action_head)                 # >>> NEW: Unified unwrap
        self.proprio_projector = _unwrap(proprio_projector)     # >>> NEW
        self.noisy_action_projector = _unwrap(noisy_action_projector)  # >>> NEW
        self.sigma_net = _unwrap(sigma_net)                     # >>> NEW


    def generate_actions(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get('micro_batch_size', batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    def generate_sequences(self, prompts):
        raise NotImplementedError(
            "HFRollout does not support generate_sequences. Use generate_actions instead."
        )
    
    def set_to_eval(self):
        self.module.eval()
        self.action_head.eval()
        self.proprio_projector.eval()
        self.noisy_action_projector.eval()
        self.sigma_net.eval()                                   # >>> NEW

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        from prismatic.training.train_utils import (
            get_current_action_mask,
            get_next_actions_mask,
        )

        # ---- inputs & masks ----
        noise   = prompts.batch["noise"]                    # a0 ~ N(0,I)  (B, chunk_len, action_dim)
        idx     = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        labels  = prompts.batch['labels']
        pixels  = prompts.batch['pixels']
        proprio = prompts.batch['proprio']

        ground_truth_token_ids = labels[:, 1:]
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask   = get_next_actions_mask(ground_truth_token_ids)

        B = idx.size(0)
        chunk_len  = noise.shape[1]
        action_dim = noise.shape[-1]
        num_patches = self.config.num_patches
        num_tokens = self.config.num_tokens

        # ---- Flow steps/time parameters (reverse integration: t 1→0, dt<0) ----
        K  = self.action_head.num_flow_steps
        dt = torch.tensor(-1.0 / K, dtype=torch.bfloat16, device=noise.device)
        t_start = torch.tensor(1.0, dtype=torch.bfloat16, device=noise.device)
        time = t_start.clone()

        # ---- Store the entire chain (for subsequent logp reproduction) ----
        x_chain = torch.empty(B, K+1, chunk_len, action_dim,
                            device=noise.device, dtype=noise.dtype)
        x_chain[:, 0] = noise

        # rollout unified eval + sampling (using sigma_net)
        self.set_to_eval()
        curr_noisy_actions = noise

        param_ctx_actor = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx_actor = FSDP.summon_full_params(self.module, writeback=False, recurse=False)

        with param_ctx_actor:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = self.module(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    pixel_values=pixels,
                    labels=labels,
                    output_hidden_states=True,
                    proprio=None,
                    proprio_projector=None,
                    noisy_actions=None,                # Write x_k back to action tokens
                    noisy_action_projector=None,
                    use_film=False,
                )

                last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
                # Get hidden states for text portion of prompt+response (after the vision patches)
                text_hidden_states = last_hidden_states[:, num_patches:-1]
                # Get hidden states for action portion of response
                actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(B, 1,num_tokens, -1).to(torch.bfloat16)
                task_latent_states = last_hidden_states[:, :num_patches].reshape(B, 1, num_patches, -1)
                all_hidden_states = torch.cat((task_latent_states, actions_hidden_states), 2)
                
                k = 0
                while time >= -dt / 2:  # Total K steps
                    
                    timesteps = torch.Tensor([1.0-time]).to(noise.device)
                    timestep_embeddings = (
                        self.action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
                    )  # (1, )

                    # Predict velocity & mean
                    flow_pred = self.action_head.predict_flow(all_hidden_states,
                                                        noisy_actions=curr_noisy_actions,
                                                        timestep_embeddings=timestep_embeddings,
                                                        noisy_action_projector=self.noisy_action_projector,
                                                        proprio=proprio,
                                                        proprio_projector=self.proprio_projector
                                                        )   # (B, chunk_len, action_dim)
                    mean_next = curr_noisy_actions + dt * flow_pred

                    # Predict std and sample (always random sampling for subsequent logp reproduction using the same chain)
                    std, _ = self.sigma_net(all_hidden_states,
                                         noisy_actions=curr_noisy_actions,
                                         timestep_embeddings=timestep_embeddings,
                                         noisy_action_projector=self.noisy_action_projector,
                                         proprio=proprio,
                                         proprio_projector=self.proprio_projector
                                         )   # (B, chunk_len, action_dim)
                    dist = torch.distributions.Normal(mean_next.to(torch.float32),
                                                    std.to(torch.float32).clamp_min(1e-6))
                    next_actions = dist.sample().to(curr_noisy_actions.dtype)

                    curr_noisy_actions = next_actions
                    x_chain[:, k+1] = curr_noisy_actions
                    time = time + dt
                    k += 1
                    if k >= K:
                        break

        predicted_actions = curr_noisy_actions

        # —— Only return rollout results and minimal set needed for logp reproduction —— 
        batch = TensorDict(
            {
                "predicted_actions": predicted_actions,          # (B, chunk_len, action_dim)
                "x_chain": x_chain,                              # (B, K+1, chunk_len, action_dim)
                # Conditions and slices needed for recomputation:
                "input_ids": idx,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixels": pixels,
                "proprio": proprio,
                "current_action_mask": current_action_mask,
                "next_actions_mask": next_actions_mask,
            },
            batch_size=B
        )

        torch.cuda.empty_cache()
        return DataProto(batch=batch)
