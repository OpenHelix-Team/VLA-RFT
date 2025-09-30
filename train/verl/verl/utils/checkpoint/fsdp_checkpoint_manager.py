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

import ray
import os
from pathlib import Path

import warnings
from typing import Union
import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from verl.utils.fs import copy_to_local, is_non_local

from transformers import PreTrainedTokenizer, ProcessorMixin

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save 
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer/processor and config for ckpt merge
    """

    def __init__(self,
                 model: FSDP,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
                 checkpoint_contents: list = ['model', 'optimizer', 'extra'],
                 **kwargs):

        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning)
            processing_class = kwargs.pop("tokenizer")
        assert "model" in checkpoint_contents and "optimizer" in checkpoint_contents and "extra" in checkpoint_contents, f"FSDPCheckpointManager must include ['model', 'optimizer', 'extra'], got {checkpoint_contents}"

        super().__init__(model,
                         optimizer,
                         lr_scheduler=lr_scheduler,
                         processing_class=processing_class,
                         checkpoint_contents=checkpoint_contents)

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        if local_path is None:
            return

        # every rank download its own checkpoint
        remote_model_path = os.path.join(local_path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt')
        remote_optim_path = os.path.join(local_path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt')
        remote_extra_state_path = os.path.join(local_path,
                                               f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt')
        print(
            f'[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}'
        )
        local_model_path = copy_to_local(remote_model_path)
        local_optim_path = copy_to_local(remote_optim_path)
        local_extra_state_path = copy_to_local(remote_extra_state_path)

        model_state_dict = torch.load(local_model_path, weights_only=False)
        optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False)

        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(
                    f'[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored'
                )

        lr_scheduler_state_dict = extra_state_dict['lr_scheduler']

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if 'rng' in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict['rng'])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(
                self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.model.state_dict()
                if self.optimizer is not None:
                    optimizer_state_dict = self.optimizer.state_dict()
                else:
                    optimizer_state_dict = None
                if self.lr_scheduler is not None:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict()
                else:
                    lr_scheduler_state_dict = None

                extra_state_dict = {
                    'lr_scheduler': lr_scheduler_state_dict,
                    'rng': self.get_rng_state(),
                }
                model_path = os.path.join(local_path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt')
                optim_path = os.path.join(local_path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt')
                extra_path = os.path.join(local_path, f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt')

                print(f'[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}')
                print(f'[rank-{self.rank}]: Saving checkpoint to {os.path.abspath(model_path)}')
                print(f'[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}')
                torch.save(model_state_dict, model_path)
                torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                torch.save(extra_state_dict, extra_path)

        if "hf_model" in self.checkpoint_contents:
            # wait for everyone to dump to local
            torch.distributed.barrier()

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, 'huggingface')
                os.makedirs(hf_local_path, exist_ok=True)
                self.model._fsdp_wrapped_module.config.save_pretrained(hf_local_path)
                self.processing_class.save_pretrained(hf_local_path)
 
        torch.distributed.barrier()

        self.previous_saved_paths.append(local_path)

class FSDPCheckpointManager_w_lora_extra_model(FSDPCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - action head
    - proprio projector
    - noisy action projector
    - optimizer
    - lr_scheduler
    - extra_states
    """
    def __init__(self,
                 model: FSDP,
                 action_head: FSDP,
                 proprio_projector: DDP,
                 noisy_action_projector: DDP,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
                 checkpoint_contents: list = ['model', 'optimizer', 'extra'],
                 **kwargs):
        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
            **kwargs
        )
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
    

    def save_checkpoint(self, local_path, hdfs_path, global_step = 0, max_ckpt_to_keep=None):
        if 'adapter' in self.checkpoint_contents:
            from torch.distributed.fsdp import FullStateDictConfig
            from peft import get_peft_model_state_dict

            if local_path is None:
                return
            # record the previous global step
            self.previous_global_step = global_step
            ckpt_name_suffix = f'{global_step}_checkpoint.pt'

            # remove previous local_path
            if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(
                    self.previous_saved_paths) >= max_ckpt_to_keep:
                keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
                self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
                self.previous_saved_paths = self.previous_saved_paths[keep_start:]
            if isinstance(local_path, str):
                local_path = Path(local_path)
            local_path = self.local_mkdir(local_path)
            adapter_path = self.local_mkdir(local_path / 'lora_adapter')
            torch.distributed.barrier()

            state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, state_dict_cfg):
                    model_state_dict = self.model._fsdp_wrapped_module.state_dict()
                # with FSDP.state_dict_type(self.action_head, StateDictType.FULL_STATE_DICT, state_dict_cfg):
                #     action_head_state_dict = self.action_head._fsdp_wrapped_module.state_dict()
                
                if self.rank == 0:
                    # adapter_state_dict = get_peft_model_state_dict(self.model._fsdp_wrapped_module, state_dict=model_state_dict)
                    # self.model._fsdp_wrapped_module.save_pretrained(adapter_path, state_dict=model_state_dict)
                    # torch.save(action_head_state_dict, local_path / f'action_head--{ckpt_name_suffix}')
                    torch.save(self.action_head.state_dict(), local_path / f'action_head--{ckpt_name_suffix}')
                    torch.save(self.noisy_action_projector.state_dict(), local_path / f'noisy_action_projector--{ckpt_name_suffix}')
                    torch.save(self.proprio_projector.state_dict(), local_path / f'proprio_projector--{ckpt_name_suffix}')
                    # torch.save(model_state_dict['base_model.model.action_queries.weight'], local_path / f'action_query--{ckpt_name_suffix}')

            torch.distributed.barrier()
            self.previous_saved_paths.append(local_path)

        else:
            super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
    def load_checkpoint(self, local_path, hdfs_path = None, del_local_after_load=False):
        if 'adapter' in self.checkpoint_contents:
            raise NotImplementedError('adapter is not supported for FSDP')
            from experiments.robot.openvla_utils import update_auto_map, check_model_logic_mismatch, _load_dataset_stats, find_checkpoint_file, load_component_state_dict
            if local_path is None:
                return
            merged_model_path = os.path.join(local_path,)


        else:
            super().load_checkpoint(local_path, hdfs_path, del_local_after_load)