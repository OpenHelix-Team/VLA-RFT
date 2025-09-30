# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple
import copy
import math

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty, agg_loss
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
import contextlib
from transformers.modeling_outputs import CausalLMOutputWithPast

__all__ = ['DataParallelPPOActor']

def _unwrap(m: nn.Module) -> nn.Module:  
    return m.module if hasattr(m, "module") else m

class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        action_head: nn.Module,
        noisy_action_projector: nn.Module,
        proprio_projector: nn.Module,
        sigma_net: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.action_head = _unwrap(action_head)
        self.noisy_action_projector = noisy_action_projector
        self.proprio_projector = proprio_projector
        self.sigma_net = _unwrap(sigma_net)
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self._is_actor = True if self.actor_optimizer is not None else False
        self.num_patches = self.config.num_patches
        self.num_tokens = self.config.num_tokens

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)  #  use torch compile by default
            else verl_F.entropy_from_logits)

    def sample_noisy_actions(self, data: DataProto) -> DataProto:
        self.action_head.eval()

        gt_actions = data.batch['gt_actions']
        with torch.no_grad():
            noise_dict = self.action_head.sample_noisy_actions(gt_actions)

        return noise_dict

    def _forward_micro_batch(self, micro_batch, return_entropy: bool = False, return_hidden_states: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Run a forward pass on a micro batch.

        Returns:
            log_probs: (B, chunk_len*action_dim)  # ← 逐维累计后的 logp
            entropy:   (B,)  (if return_entropy is True)
        """
        x_chain = micro_batch['x_chain']                          # (B, K+1, chunk_len, action_dim)
        B, Kp1, chunk_len, action_dim = x_chain.shape
        K = Kp1 - 1
        assert K > 0, "x_chain len must be > 1"

        input_ids       = micro_batch['input_ids']
        attention_mask  = micro_batch['attention_mask']
        labels          = micro_batch['labels']
        pixels          = micro_batch['pixels']
        proprio         = micro_batch['proprio']
        current_mask    = micro_batch['current_action_mask']
        next_mask       = micro_batch['next_actions_mask']

        # Backward integration step
        dt = -1.0 / K

        device = input_ids.device
        # 逐维累计的 logp（float32 累加更稳）
        logp_per_dim = torch.zeros(B, chunk_len, action_dim, device=device, dtype=torch.float32)
        entropy_per_dim   = torch.zeros(B, chunk_len, action_dim, device=device, dtype=torch.float32)
        const_term = 0.5 * (torch.log(torch.tensor(2.0 * torch.pi, device=device, dtype=torch.float32)) + 1.0)  # 0.5*log(2πe)

        # ---- Encode context once ----
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixels,
                labels=labels,
                output_hidden_states=True,
                proprio=None,
                proprio_projector=None,
                noisy_actions=None,                
                noisy_action_projector=None,
                use_film=False,
            )
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            text_hidden_states = last_hidden_states[:, self.num_patches:-1]
            actions_hidden_states = text_hidden_states[current_mask | next_mask].reshape(
                B, 1, self.num_tokens, -1
            ).to(torch.bfloat16)  # (B, 1, T_actions, H)
            task_latent_states = last_hidden_states[:, :self.num_patches].reshape(
                B, 1, self.num_patches, -1
            )  # (B, 1, T_vision, H)
            all_hidden_states = torch.cat((task_latent_states, actions_hidden_states), dim=2)  # (B, 1, T_ctx, H)

        # ---- Reproduce the rollout chain step-by-step ----
        for k in range(K):
            x_k  = x_chain[:, k]     # (B, chunk_len, action_dim)
            x_k1 = x_chain[:, k+1]   # (B, chunk_len, action_dim)

            # Linear time schedule: t_k = k / K
            t_scalar = (k / K)
            t_embed = torch.tensor([[t_scalar]], device=device, dtype=x_k.dtype)  # (1, 1)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # flow & std 与生成时同一套输入
                flow = self.action_head.predict_flow(
                    all_hidden_states,
                    noisy_actions=x_k,
                    timestep_embeddings=t_embed,
                    noisy_action_projector=self.noisy_action_projector,
                    proprio=proprio,
                    proprio_projector=self.proprio_projector,
                )   # (B, chunk_len, action_dim)

                std, log_std = self.sigma_net(
                    all_hidden_states,
                    noisy_actions=x_k,
                    timestep_embeddings=t_embed,
                    noisy_action_projector=self.noisy_action_projector,
                    proprio=proprio,
                    proprio_projector=self.proprio_projector,
                )   # (B, chunk_len, action_dim)

            mean_next = x_k + dt * flow

            # per-dim logp
            dist = torch.distributions.Normal(
                mean_next.to(torch.float32),
                std.to(torch.float32).clamp_min(1e-6)
            )
            step_logp = dist.log_prob(x_k1.to(torch.float32))  # (B, chunk_len, action_dim)
            logp_per_dim += step_logp

            if return_entropy:
                # per-dim entropy 累计：log σ + 0.5*log(2πe)
                entropy_per_dim += log_std.to(torch.float32) + const_term              # (B, chunk_len, action_dim)
        # breakpoint()
        # 展平成 (B, chunk_len*action_dim)
        logp_vec = logp_per_dim.reshape(B, chunk_len * action_dim).to(torch.bfloat16)
        if return_entropy:
            entropy_per_dim = entropy_per_dim / (K + 1)  
            entropy_vec = entropy_per_dim.reshape(B, chunk_len * action_dim).to(torch.bfloat16)
            if return_hidden_states:
                # breakpoint()
                return logp_vec, entropy_vec, all_hidden_states
            else:
                return logp_vec, entropy_vec
        else:
            return logp_vec

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        max_norm = float(self.config.grad_clip)

        # ---- helpers ----
        def has_nonfinite_grads(module, name):
            for p in module.parameters(recurse=True):
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    # 打印一个示例，便于快速定位
                    with torch.no_grad():
                        g = p.grad
                        print(f"[Nonfinite] {name}: param shape={tuple(p.shape)} "
                            f"min={g.min().item() if torch.isfinite(g).any() else 'nan'} "
                            f"max={g.max().item() if torch.isfinite(g).any() else 'nan'} "
                            f"mean={(g[g.isfinite()].mean().item() if g.isfinite().any() else 'nan')}")
                    return True
            return False

        def clip_one(module, name):
            """返回 (norm, ok)；ok=False 表示非有限"""
            if module is None:
                return 0.0, True
            # 先做非有限检查，能更早定位问题
            if has_nonfinite_grads(module, name):
                return float("nan"), False

            try:
                if isinstance(module, FSDP):
                    n = module.clip_grad_norm_(max_norm=max_norm)  # FSDP 自带全局规约
                    n = float(n) if not isinstance(n, float) else n
                else:
                    n = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=max_norm,
                                                    error_if_nonfinite=False)
                    n = float(n)
            except Exception as e:
                print(f"[ClipError] {name}: {e}")
                return float("nan"), False

            if not math.isfinite(n):
                print(f"[ClipNonfinite] {name}: total_norm={n}")
                return n, False
            return n, True

        # ---- 要裁剪的模块列表（名称仅用于日志）----
        modules = []
        # modules.append(("actor", self.actor_module))
        modules.append(("action_head", getattr(self, "action_head", None)))
        modules.append(("sigma_net", getattr(self, "sigma_net", None)))
        if getattr(self, "proprio_projector", None) is not None:
            modules.append(("proprio_projector", self.proprio_projector))
        if getattr(self, "noisy_action_projector", None) is not None:
            modules.append(("noisy_action_projector", self.noisy_action_projector))


        if not isinstance(self.actor_module, FSDP):
            raise NotImplementedError("clip_grad_norm_ is not implemented for non-FSDP actor")

        # ---- 逐组裁剪 + 合成全局范数 ----
        total_sq = 0.0
        all_ok = True
        per_group = {}
        for name, m in modules:
            n, ok = clip_one(m, name)
            per_group[name] = n
            all_ok &= ok
            if math.isfinite(n):
                total_sq += (n * n)

        # 最后的“全局”L2 范数（各不相交参数组的 L2 合成）
        global_norm = float(math.sqrt(total_sq)) if math.isfinite(total_sq) else float("nan")

        if not all_ok or not math.isfinite(global_norm):
            print(f"WARN: grad_norm is not finite. per_group={per_group}, global={global_norm}")
            self.actor_optimizer.zero_grad(set_to_none=True)
            return float("nan")

        # 一切正常，执行优化步
        self.actor_optimizer.step()
        return global_norm

    # def _optimizer_step(self):
    #     assert self.config.grad_clip is not None

    #     if isinstance(self.actor_module, FSDP):
    #         grad_norms = []
    #         grad_norms.append(self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip))
    #         grad_norms.append(torch.nn.utils.clip_grad_norm_(self.action_head.parameters(), max_norm=self.config.grad_clip))
    #         grad_norms.append(torch.nn.utils.clip_grad_norm_(self.sigma_net.parameters(), max_norm=self.config.grad_clip))
    #         if self.proprio_projector is not None:
    #             grad_norms.append(torch.nn.utils.clip_grad_norm_(self.proprio_projector.parameters(), max_norm=self.config.grad_clip))
    #         if self.noisy_action_projector is not None:
    #             grad_norms.append(torch.nn.utils.clip_grad_norm_(self.noisy_action_projector.parameters(), max_norm=self.config.grad_clip))
    #         grad_norm = max(grad_norms)
    #     else:
    #         raise NotImplementedError("clip_grad_norm_ is not implemented for non-FSDP actor")
    #         grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

    #     # if grad_norm is not finite, skip the update
    #     if not torch.isfinite(grad_norm):
    #         print(f"WARN: grad_norm is not finite: {grad_norm}")
    #         self.actor_optimizer.zero_grad()
    #     else:
    #         self.actor_optimizer.step()
    #     return grad_norm

    def _set_to_eval(self):
        self.actor_module.eval()
        self.action_head.eval()
        self.proprio_projector.eval()
        self.noisy_action_projector.eval()
        self.sigma_net.eval()
    
    def _set_to_train(self):
        assert self._is_actor, "set_to_train should only be called for actor not reference policy"
        self.actor_module.train()
        self.action_head.train()
        self.proprio_projector.train()
        self.noisy_action_projector.train()
        self.sigma_net.train()

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``x_chain``: tensor of shape [batch_size, K+1, chunk_len, action_dim]. torch.float32/bfloat16.
                The action trajectory chain used for flow matching, where K is the number of flow steps.

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64. Mask to avoid
                performing attention on padding token indices.

                ``labels``: tensor of shape [batch_size, sequence_length]. torch.int64. Labels for computing
                the masked language modeling loss.

                ``pixels``: tensor of shape [batch_size, num_patches, patch_dim]. torch.float32/bfloat16.
                Visual input features from image patches.

                ``proprio``: tensor of shape [batch_size, proprio_dim]. torch.float32/bfloat16.
                Proprioceptive sensor data (e.g., robot joint states, gripper status).

                ``current_action_mask``: tensor of shape [batch_size, sequence_length]. torch.bool.
                Mask indicating which tokens correspond to current action predictions.

                ``next_actions_mask``: tensor of shape [batch_size, sequence_length]. torch.bool.
                Mask indicating which tokens correspond to next action predictions.

            data.meta_info contains:

                ``micro_batch_size`` (int): Size of each micro batch for processing.

                ``use_dynamic_bsz`` (bool): Whether to use dynamic batch sizing based on token length.

                ``max_token_len`` (int, optional): Maximum token length per micro batch when using dynamic sizing.

        Returns:
            torch.Tensor: Log probability tensor of shape [batch_size]. The log probability of the entire
            action sequence for each sample in the batch, computed via flow matching dynamics.
        """
        # set to eval
        self._set_to_eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = [
            'x_chain',
            'input_ids', 'attention_mask', 'labels', 'pixels', 'proprio',
            'current_action_mask', 'next_actions_mask',
        ]
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                lp = self._forward_micro_batch(micro_batch, return_entropy=False)
            log_probs_lst.append(lp)

        log_probs = torch.concat(log_probs_lst, dim=0).to(torch.bfloat16)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self._set_to_train()
        # breakpoint()
        select_keys = ['x_chain', 'advantages', 'attention_mask', 'current_action_mask', 'input_ids', 'labels',
                       'next_actions_mask', 'old_log_probs', 'pixels', 'predicted_actions', 'proprio']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_probs')
        if self.config.use_mse_loss or self.config.log_mse_loss:
            select_keys.extend(['flow', 'gt_noisy_actions', 'gt_timestep_embeddings'])
        if self.config.log_l1_loss:
            select_keys.extend(['gt_actions', 'predicted_actions'])
        batch = data.select(batch_keys=select_keys).batch
        # breakpoint()  # for debugging
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)
        
        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    assert self.gradient_accumulation >= 1, "ppo_mini_batch_size must be >= ppo_micro_batch_size_per_gpu"
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                
                for data in micro_batches:
                    # # Support all hardwares
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        old_log_probs = data['old_log_probs']
                        advantages = data['advantages']

                        dummy_response_mask = torch.ones_like(advantages)

                        clip_ratio = self.config.clip_ratio
                        clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                        clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                        clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
                        entropy_coeff = self.config.entropy_coeff
                        loss_agg_mode = self.config.loss_agg_mode


                        # all return: (bsz, response_length)
                        if self.config.use_mse_loss:
                            new_log_probs, entropy, all_hidden_states= self._forward_micro_batch(micro_batch=data, return_entropy=True, return_hidden_states=True)
                        else:
                            new_log_probs, entropy= self._forward_micro_batch(micro_batch=data, return_entropy=True)
                        # breakpoint()

                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_probs,
                            log_prob=new_log_probs,
                            advantages=advantages,
                            response_mask=dummy_response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            log_prob_aggregated=False)
                        
                        # compute entropy loss from entropy
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=dummy_response_mask, loss_agg_mode=loss_agg_mode)
                        
                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                        if self.config.log_l1_loss:
                            with torch.no_grad():
                                gt_actions = data['gt_actions']
                                predicted_actions = data['predicted_actions']
                                l1_loss = nn.functional.l1_loss(predicted_actions, gt_actions, reduction="mean")
                                metrics['actor/l1_loss'] = l1_loss.detach().item()

                        if self.config.use_mse_loss:
                            # self.config.log_mse_loss=False
                            # with torch.no_grad():
                            with torch.no_grad():
                                t = (ppo_kl - self.config.mse_kl_low) / (self.config.mse_kl_high - self.config.mse_kl_low)
                                gate = torch.clamp(t, 0.0, 1.0)      # [0,1]
                                mse_loss_coef = self.config.mse_loss_coef * gate
                            if mse_loss_coef > 0:
                                flow             = data['flow']
                                proprio          = data['proprio']
                                gt_noisy_actions = data['gt_noisy_actions']
                                gt_timestep_embeddings = data['gt_timestep_embeddings']
                        
                                # Velocity and std
                                flow_pred = self.action_head.predict_flow(all_hidden_states,
                                                                        noisy_actions=gt_noisy_actions,
                                                                        timestep_embeddings=gt_timestep_embeddings,
                                                                        noisy_action_projector=self.noisy_action_projector,
                                                                        proprio=proprio,
                                                                        proprio_projector=self.proprio_projector)        # (B, chunk_len, action_dim)
                                flow_pred = flow_pred.reshape(flow.shape)
                                mse_loss = nn.functional.mse_loss(flow_pred, flow, reduction="mean")
                                policy_loss = policy_loss + mse_loss * mse_loss_coef
                                metrics['actor/mse_loss'] = mse_loss.detach().item()
                                metrics['actor/mse_coef'] = mse_loss_coef.detach().item()
                        
                        if self.config.use_kl_loss:
                            ref_log_probs = data['ref_log_probs']
                            # compute kl loss
                            kld = kl_penalty(logprob=new_log_probs,
                                            ref_logprob=ref_log_probs,
                                            kl_penalty=self.config.kl_loss_type)
                            dummy_kl_mask = torch.ones_like(kld)
                            kl_loss = agg_loss(loss_mat=kld,
                                            loss_mask=dummy_kl_mask,
                                            loss_agg_mode=self.config.loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics['actor/kl_loss'] = kl_loss.detach().item()
                            metrics['actor/kl_coef'] = self.config.kl_loss_coef
                        # breakpoint()
                        if self.config.use_dynamic_bsz:
                            raise NotImplementedError("Dynamic batch size is not supported in DataParallelPPOActor.")
                            # relative to the dynamic bsz
                            # loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation
                            # loss = ce_loss / self.gradient_accumulation

                        # check_params_full(self.actor_module)  # check if all params are full after summon
                        
                        loss.backward()

                        data = {
                            'actor/entropy': entropy_loss.detach().item(),
                            'actor/pg_loss': pg_loss.detach().item(),
                            'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                            'actor/ppo_kl': ppo_kl.detach().item(),
                            'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item(),
                        }
                        append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm}
                # breakpoint()
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
