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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from collections import defaultdict
from functools import partial
from tqdm import tqdm

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, bootstrap_metric, calc_maj_val, process_validation_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Tokenizer = 7
    WorldModelRollout = 8


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'actor/reward_kl_penalty': current_kl, 'actor/reward_kl_penalty_coeff': beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]

def compute_dummy_response_mask(data: DataProto):
    dummy_mask = torch.ones([data.batch['token_level_rewards'].shape[0], 8*7])
    return dummy_mask.to(data.batch['token_level_rewards'].device)

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, uniform_std=False):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch['response_mask'] = compute_dummy_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch['token_level_rewards'],
            values=data.batch['values'],
            response_mask=data.batch['response_mask'],
            gamma=gamma,
            lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'],
            uniform_std=uniform_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            eos_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            reward_baselines=data.batch['reward_baselines'],
            response_mask=data.batch['response_mask'])

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size", 
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",  # TODO: maybe modifeid
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                         config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                         "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")
        

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                                         num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.get('gen_batch_size',
                                                                                   self.config.data.train_batch_size),
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.get('truncation', 'error'),
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                                       num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            if "reward_extra_info" in result:
                reward_extra_infos_dict["final_reward"].extend(scores)
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for lst in reward_extra_infos_dict.values():
            assert len(lst) == 0 or len(lst) == len(sample_scores)

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)

        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "final_reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if var_name == core_var and any(
                            metric_name.startswith(pfx)
                            for pfx in ["mean", "std", "maj", "best"]) and f"@{n_max}/" in metric_name:
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self, save_dataloader=True, save_data_statistics=False):
        from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        if save_data_statistics:
            data_local_path = os.path.join(local_global_step_folder, 'data')
            save_dataset_statistics(self.train_dataset.dataset_statistics, data_local_path)

        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')

        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep',
                                                         None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get('max_critic_ckpt_to_keep',
                                                          None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        if save_dataloader:
            dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch['response_mask'] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        raise NotImplementedError
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result['reward_extra_info']
                        except Exception as e:
                            print(f'Error in reward_fn: {e}')
                            reward_tensor = self.reward_fn(batch)
                            reward_extra_infos_dict = {}

                        batch.batch['token_level_scores'] = reward_tensor

                        print(f'{list(reward_extra_infos_dict.keys())=}')
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl_in_reward,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1


from ivideogpt.data import DATASET_NAMED_MIXES, SimpleRoboticDataLoaderv2, EvalDataLoader
from ivideogpt.tokenizer import TOKENIZER
import cv2
import pickle
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform_V1
from torch.utils.data import DataLoader
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

LOSS_KEYS = ['lpips', 'mse', 'mae', 'ssim', 'psnr']




def print_batch(batch):
    for i in range(10):
        print(batch.batch["input_ids"][i, :10])


class RayMARVELPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        # instantiate tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.use_wm = Role.WorldModelRollout in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()



    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError
        
        if self.use_wm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.WorldModelRollout)
            wm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.WorldModelRollout],
                                         config=self.config.world_model_rollout,
                                         role='wm_rollout')
            self.resource_pool_to_cls[resource_pool]['wm_rollout'] = wm_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # create tokenizer
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Tokenizer)
        tok_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Tokenizer], config=self.config.processor)
        self.resource_pool_to_cls[resource_pool]['tok'] = tok_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        self.tokenizer_wg = all_wg['tok']
        self.tokenizer_wg.init_model()

        if self.use_wm:
            self.wm_rollout_wg = all_wg['wm_rollout']
            self.wm_rollout_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

        self._create_dataloader()

    def _create_dataloader(self):
        config = self.config.data.video
        #  TODO: processor should keep unchanged during training and inference
        if self.use_reference_policy:
            processor_list = self.ref_policy_wg.get_processor()
            processor = processor_list[0]
        else:
            processor_list = self.actor_rollout_wg.get_processor()
            processor = processor_list[0]
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        batch_transform = RLDSBatchTransform_V1(
            action_tokenizer,
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=False,
            use_proprio=True,
            use_minivla=True,
            use_raw_image=self.config.data.use_raw_image
        )
        self.train_dataset = RLDSDataset(
            config.dataset_path,
            config.dataset_name,
            batch_transform,
            resize_resolution=tuple(config.resolution),
            shuffle_buffer_size=self.config.data.shuffle_buffer_size,
            image_aug=self.config.data.image_aug,
        )
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
            )
        
        self.val_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
            )

        self.total_training_steps = self.config.trainer.total_training_steps
        print(f'Total training steps: {self.total_training_steps}')
    def plot_img(self, img, postfix='', reward=None):
        img_np = (img[0].detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype('uint8').copy()
        reward_str = f"_reward{reward:.4f}" if reward is not None else ""
        filename = f'{self.config.trainer.img_save_dir}/{self.global_steps}_tmp-img{postfix}{reward_str}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if reward is not None:
            cv2.putText(img_np, f"Reward: {reward:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(filename, img_np)
    def reward_fn(self, batch: DataProto, pixels, return_reward_tensor=True, save_pred=False, pixels_before_repeat=None):
        if pixels_before_repeat is None:
            pixels_before_repeat = pixels

        output_tokens = batch.batch["responses"][:, -self.config.processor.tokens_per_frame - 1:-1]
        output_tokens = output_tokens.clamp(0, self.config.processor.visual_token_num - 1).long()
        if "ctx_tokens" in batch.batch.keys():
            ctx_tokens = batch.batch["ctx_tokens"]
            output_tokens = output_tokens.reshape(*output_tokens.shape[:-1], 1, 80)  # TODO: magic number
            detokenize_output = self.tokenizer_wg.detokenize(
                DataProto.from_single_dict({"tokens": output_tokens, "ctx_tokens": ctx_tokens}),
                DataProto.from_single_dict({"real": pixels_before_repeat[:, -1]}, meta_info={'lpips': True, 'loss_weight': self.config.trainer.loss_weight}))
            pred = detokenize_output.batch['pixels'][:, -1]
        else:
            output_tokens = output_tokens.reshape(*output_tokens.shape[:-1], 16, 20)  # TODO: magic number
            detokenize_output = self.tokenizer_wg.detokenize(
                DataProto.from_single_dict({"tokens": output_tokens}),
                DataProto.from_single_dict({"real": pixels_before_repeat[:, -1]}, meta_info={'lpips': True, 'loss_weight': self.config.trainer.loss_weight}))
            pred = detokenize_output.batch['pixels']
        pred = pred.clamp(0.0, 1.0)
        real = pixels[:, -1]
        input = pixels[:, -2]
        
        if 'loss' in detokenize_output.batch.keys():
            loss = detokenize_output.batch['loss']
        else:
            if self.config.trainer.reward_fn == 'mse':
                recon_loss = torch.mean((real - pred)**2, dim=(1, 2, 3))
            elif self.config.trainer.reward_fn == 'mae':
                recon_loss = torch.mean(torch.abs(real - pred), dim=(1, 2, 3))
            else:
                raise NotImplementedError(f"Unsupported reward function: {self.config.trainer.reward_fn}")
            # recon_loss = self.tokenizer_wg.recon_loss(DataProto.from_single_dict({"real": real, "pred": pred}))
            # recon_loss = recon_loss.batch['recon_loss']
            
            if 'perceptual_loss' in detokenize_output.batch.keys():
                perceptual_loss = detokenize_output.batch['perceptual_loss']
            else:
                perceptual_loss = self.tokenizer_wg.perceptual_loss(DataProto.from_single_dict({"real": real, "pred": pred}))
                perceptual_loss = perceptual_loss.batch['perceptual_loss']
            
            loss = recon_loss + perceptual_loss

        if not return_reward_tensor:
            return loss

        print("loss", loss.mean().item(), flush=True)
        if 'loss' in detokenize_output.batch.keys():
            for k in LOSS_KEYS:
                if k in detokenize_output.batch.keys():
                    print(f"{k}", detokenize_output.batch[k].mean().item(), flush=True)
        else:
            print("recon_loss", recon_loss.mean().item(), flush=True)
            print("perceptual_loss", perceptual_loss.mean().item(), flush=True)
        if save_pred:
            for i in range(5):
                self.plot_img(torch.cat([
                    input[i:i+1].float(),
                    real[i:i+1].float(),
                    pred[i:i+1].float(),
                    torch.abs(real[i:i+1].float()-pred[i:i+1].float()),
                ], dim=-2), f"{self.config.trainer.experiment_name}-{i}")
            

        reward_tensor = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)
        for i in range(len(batch)):
            data_item = batch[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().long().item()
            reward_tensor[i, valid_response_length - 1] = -loss[i].item()

        if 'loss' in detokenize_output.batch.keys():
            return reward_tensor, {
                f"critic/{k}/mean": detokenize_output.batch[k].mean().item()
                for k in LOSS_KEYS if k in detokenize_output.batch.keys()
            }
        else:
            return reward_tensor, {
                "critic/recon_loss/mean": recon_loss.mean().item(),
                "critic/perceptual_loss/mean": perceptual_loss.mean().item(),
            }
        
    def msp_reward_fn(self, batch: DataProto, pixels, return_reward_tensor=True, save_pred=False, pixels_before_repeat=None):
        if pixels_before_repeat is None:
            pixels_before_repeat = pixels
        
        batch_size = batch.batch["responses"].shape[0]
        segment_length = self.config.data.video.segment_length
        tokens_per_frame = self.config.processor.tokens_per_frame
        action_dim = self.config.processor.action_dim
        # breakpoint()
        output_tokens = batch.batch["responses"].reshape(batch_size, segment_length - 1, tokens_per_frame + action_dim)
        output_tokens = output_tokens[:, :, :tokens_per_frame]
        output_tokens = output_tokens.clamp(0, self.config.processor.visual_token_num - 1).long()
        
        ctx_tokens = batch.batch["ctx_tokens"]
        output_tokens = output_tokens.reshape(batch_size, segment_length - 1, tokens_per_frame)

        if self.config.world_model_rollout.rollout.w_gt_ac:
            gt_output_tokens = batch.batch["gt_responses"].reshape(batch_size, segment_length - 1, tokens_per_frame + action_dim)
            gt_output_tokens = gt_output_tokens[:, :, :tokens_per_frame]
            gt_output_tokens = gt_output_tokens.clamp(0, self.config.processor.visual_token_num - 1).long()
            detokenize_output = self.tokenizer_wg.detokenize(
                DataProto.from_single_dict({"tokens": output_tokens, "ctx_tokens": ctx_tokens}),
                # DataProto.from_single_dict({"real": pixels_before_repeat[:, 2:]}, meta_info={'lpips': True}),
                DataProto.from_single_dict({"real": gt_output_tokens}, meta_info={'lpips': True, 'recon': self.config.trainer.reward_fn}),
            )
        else:
            detokenize_output = self.tokenizer_wg.detokenize(
                DataProto.from_single_dict({"tokens": output_tokens, "ctx_tokens": ctx_tokens}),
                DataProto.from_single_dict({"dummy": torch.zeros((batch_size, 1))}, meta_info={'lpips': True, 'recon': self.config.trainer.reward_fn}),
            )

        if 'recon_loss' in detokenize_output.batch.keys():
            recon_loss = detokenize_output.batch['recon_loss']
        else:
            pred = detokenize_output.batch['pixels'].clamp(0.0, 1.0)[:, 1:]
            real = pixels[:, 2:]
            if self.config.trainer.reward_fn == 'mse':
                recon_loss = torch.mean((real - pred)**2, dim=(2, 3, 4))
            elif self.config.trainer.reward_fn == 'mae':
                recon_loss = torch.mean(torch.abs(real - pred), dim=(2, 3, 4))
            else:
                raise NotImplementedError(f"Unsupported reward function: {self.config.trainer.reward_fn}")
        
        if 'perceptual_loss' in detokenize_output.batch.keys():
            perceptual_loss = detokenize_output.batch['perceptual_loss']
        else:
            perceptual_loss = self.tokenizer_wg.perceptual_loss(DataProto.from_single_dict({"real": real, "pred": pred}))
            perceptual_loss = perceptual_loss.batch['perceptual_loss']
        
        mse_weight = getattr(self.config.trainer.loss_weight, self.config.trainer.reward_fn, 1.0)
        perceptual_weight = getattr(self.config.trainer.loss_weight, 'lpips', 1.0)
        
        if self.config.trainer.msp_reward_aggregate == "mean":
            loss = (recon_loss * mse_weight + perceptual_loss * perceptual_weight).mean(-1)
        elif self.config.trainer.msp_reward_aggregate == "last":
            loss = (recon_loss * mse_weight + perceptual_loss * perceptual_weight)[:, -1]
        elif self.config.trainer.msp_reward_aggregate == "discount":
            discount = self.config.trainer.msp_reward_discount
            weight = discount ** torch.arange(recon_loss.shape[1] - 1, -1, -1, device=recon_loss.device)
            # breakpoint()
            loss = (recon_loss * mse_weight + perceptual_loss * perceptual_weight) * weight.unsqueeze(0)
            loss = loss.sum(-1) / weight.sum()
        if not return_reward_tensor:
            return loss

        # print("loss", loss.mean().item(), flush=True)
        # print("recon_loss", recon_loss.mean().item(), flush=True)
        # print("perceptual_loss", perceptual_loss.mean().item(), flush=True)
        if save_pred:
            # breakpoint()
            pred = detokenize_output.batch['pixels'].clamp(0.0, 1.0)[:, 1:]
            real = pixels.batch['pixels'][:, 2:]
            if self.config.world_model_rollout.rollout.w_gt_ac:
                gt_ac_gen = detokenize_output.batch['real']
                for i in range(len(batch)):
                # for i in range(1):
                    real_traj = torch.cat([real[i:i+1, j] for j in range(real.shape[1])], dim=-1)
                    pred_traj = torch.cat([pred[i:i+1, j] for j in range(pred.shape[1])], dim=-1)
                    gt_ac_gen_traj = torch.cat([gt_ac_gen[i:i+1, j] for j in range(gt_ac_gen.shape[1])], dim=-1)
                    # breakpoint()
                    self.plot_img(torch.cat([
                        real_traj.float(),
                        gt_ac_gen_traj.float(),
                        pred_traj.float(),
                        torch.abs(real_traj-pred_traj).float(),
                    ], dim=-2), f"{self.config.trainer.experiment_name}-{i}", -loss[i].item())
            else:
                for i in range(len(batch)):
                    real_traj = torch.cat([real[i:i+1, j] for j in range(real.shape[1])], dim=-1)
                    pred_traj = torch.cat([pred[i:i+1, j] for j in range(pred.shape[1])], dim=-1)
                    self.plot_img(torch.cat([
                        real_traj.float(),
                        pred_traj.float(),
                        torch.abs(real_traj-pred_traj).float(),
                    ], dim=-2), f"{self.config.trainer.experiment_name}-{i}", -loss[i].item())

        reward_tensor = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)

        # baseline_loss = getattr(self.config.trainer, 'baseline_loss', 0.0)
        # reward_weight = getattr(self.config.trainer, 'reward_weight', 1.0)
        for i in range(len(batch)):
            data_item = batch[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().long().item()
            # reward_tensor[i, valid_response_length - 1] = (baseline_loss -loss[i].item()) * reward_weight
            reward_tensor[i, valid_response_length - 1] = -loss[i].item()
        return reward_tensor, {
            "critic/recon_loss/mean": recon_loss.mean().item(),
            "critic/perceptual_loss/mean": perceptual_loss.mean().item(),
        }
    
    def ac_reward_fn(self,
                    batch: DataProto,
                    return_reward_tensor: bool = True,
                    reward_type: str = 'l1',
                    huber_delta: float = 1.0):
        """
        Compute action-wise rewards from differences between predicted and ground-truth actions.

        Args:
            batch: DataProto with keys:
                - 'gt_actions'         : [bs, chunk_len, ac_dim]
                - 'predicted_actions'  : [bs, chunk_len, ac_dim]
            return_reward_tensor: If True, return element-wise rewards with shape [bs, chunk_len * ac_dim];
                                otherwise return per-sample mean reward [bs].
            reward_type: One of {'l1', 'mse', 'huber'}.
            huber_delta: Delta parameter for Huber loss (a.k.a. Smooth L1). Controls the transition
                        between quadratic and linear regions.

        Returns:
            Tensor:
                - If return_reward_tensor=True  -> [bs, chunk_len * ac_dim]
                - Else                          -> [bs]
        """
        gt = batch.batch["gt_actions"]
        pred = batch.batch["predicted_actions"]

        bs = gt.shape[0]
        # Flatten to [bs, chunk_len * ac_dim] for element-wise reward
        gt_flat   = gt.reshape(bs, -1).to(dtype=torch.float32)
        pred_flat = pred.reshape(bs, -1).to(dtype=torch.float32)

        diff = pred_flat - gt_flat
        abs_diff = diff.abs()

        if reward_type == 'l1':
            # Element-wise L1 loss
            loss_tensor = abs_diff
        elif reward_type == 'mse':
            # Element-wise squared error
            loss_tensor = diff ** 2
        elif reward_type == 'huber':
            # Huber (Smooth L1): quadratic near 0, linear for large residuals
            # 0.5 * x^2                      if |x| <= delta
            # delta * (|x| - 0.5 * delta)    otherwise
            delta = huber_delta
            loss_tensor = torch.where(
                abs_diff <= delta,
                0.5 * diff**2,
                delta * (abs_diff - 0.5 * delta)
            )
        else:
            raise ValueError(f"Unsupported reward_type: {reward_type}")

        # Define reward as negative loss so that smaller error => larger reward
        reward_tensor = -loss_tensor

        if return_reward_tensor:
            # Element-wise rewards: [bs, chunk_len * ac_dim]
            return reward_tensor, {
                f"critic/{reward_type}_loss/mean": loss_tensor.mean().item(),
            }
        else:
            # Per-sample mean reward: [bs]
            return reward_tensor.mean(dim=1), {
                f"critic/{reward_type}_loss/mean": loss_tensor.mean().item(),
            }

    def _validate(self):
        # Lists to collect samples for the table
        sample_scores = []
        val_losses = defaultdict(list)

        val_iters = 0
        for batch in tqdm(self.val_dataloader, total=self.config.trainer.val_iters, desc=f"validation"):
            val_iters += 1

            pixels, gt_actions, instructions = batch
            ac_batch = DataProto.from_single_dict({"pixels": pixels, "instructions": instructions})
            ac_batch = self.actor_rollout_wg.generate_actions(ac_batch)  # old_actions, new_actions, ref_actions
            wm_batch = DataProto.from_single_dict({"pixels": pixels, "gt_actions": gt_actions})
            wm_batch.union(ac_batch)
            batch = self.tokenizer_wg.process(wm_batch)
            if 'ctx_tokens' in batch.batch.keys():
                ctx_tokens = batch.pop(batch_keys=['ctx_tokens'])
            else:
                ctx_tokens = None
            batch = DataProto.from_single_dict({
                k: v[:, :self.config.processor.gen_input_length] for k, v in batch.batch.items()
            })

            # pop those keys for generation
            if self.config.actor_rollout_ref.rollout.interact:
                test_gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'action_ids'])
            else:
                test_gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.config.processor.eos_token_id,
                'pad_token_id': self.config.processor.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }

            test_gen_batch_output = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            if ctx_tokens is not None:
                test_gen_batch_output = test_gen_batch_output.union(ctx_tokens)

            reward_fn = self.msp_reward_fn if self.config.processor.processor_type == 'ctx_msp' else self.reward_fn
            rewards, losses = reward_fn(test_gen_batch_output, pixels, return_reward_tensor=True)
            sample_scores += rewards.sum(-1).cpu().tolist()
            for k, v in losses.items():
                val_losses[k].append(v)

            if val_iters == self.config.trainer.val_iters:
                break

        metric_dict = {
            "val/reward": sum(sample_scores) / len(sample_scores),
            **{"val/" + k: sum(v) / len(v) for k, v in val_losses.items()},
        }
        return metric_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch in self.train_dataloader:
                # breakpoint()
                pixel_values = batch['pixel_values']
                proprio = batch['proprio']
                actor_input_ids = batch['input_ids']
                actor_attention_mask = batch['attention_mask']
                actor_labels = batch['labels']
                gt_actions = batch['actions']  
                raw_pixel_values = batch['raw_pixel_values']
                # breakpoint()
                actor_batch = DataProto.from_single_dict({
                    "pixels": pixel_values, 
                    "proprio": proprio,
                    "input_ids": actor_input_ids, 
                    "attention_mask": actor_attention_mask,
                    "labels": actor_labels, 
                    "gt_actions": gt_actions
                })

                wm_batch = DataProto.from_single_dict({
                    "pixels": raw_pixel_values
                })
                
                if self.config.world_model_rollout.rollout.w_gt_ac:
                    wm_batch.batch['gt_actions'] = gt_actions

                metrics = {}
                timing_raw = {}

                is_last_step = self.global_steps >= self.total_training_steps
                
                with _timer('step', timing_raw):

                    actor_gen_batch = actor_batch.pop(batch_keys=['pixels', 'proprio', 
                                                                'input_ids', 'attention_mask', 'labels'])
                    
                    # noise: (batch_size*n, T, ac_dim) : sample for rollout
                    # gt_noise_actions, gt_diffusion_timestepembeddings : sample for train mse loss
                    noise_batch = self.actor_rollout_wg.sample_noisy_actions(actor_batch)  
                    actor_gen_batch = actor_gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    noise_batch_partial = noise_batch.pop(batch_keys=['noise'])
                    actor_gen_batch = actor_gen_batch.union(noise_batch_partial)
                    # breakpoint()
                    # rollout actions
                    with _timer('ac_rollout', timing_raw):
                        actor_batch_output = self.actor_rollout_wg.generate_actions(actor_gen_batch)  
                        actor_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(actor_batch.batch))],
                                                                    dtype=object)
                        # repeat to align with repeated responses in rollout
                        actor_batch = actor_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        actor_batch = actor_batch.union(actor_batch_output)
                        actor_batch = actor_batch.union(noise_batch)

                    # breakpoint()
                    # recompute old_log_probs
                    with _timer('log_prob', timing_raw):
                        log_prob = self.actor_rollout_wg.compute_log_prob(actor_batch_output) 
                        if self.use_reference_policy:
                            log_prob_ref = self.ref_policy_wg.compute_ref_log_prob(actor_batch_output) 
                            log_prob = log_prob.union(log_prob_ref)
                        # input: ['responses', 'input_ids', 'attention_mask', 'position_ids']  output: {'old_log_prob':..., 'ref_log_prob':..., 'new_log_prob'}] [bs, response_len] 
                        actor_batch = actor_batch.union(log_prob)

                    wm_batch = wm_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    wm_batch = wm_batch.union(actor_batch_output.pop(batch_keys=['predicted_actions']))
                    
                    if self.config.trainer.use_ac_reward:
                        with _timer('ac_reward', timing_raw):
                            reward_fn = self.ac_reward_fn

                            reward_tensor, losses = reward_fn(batch = actor_batch,
                                                              reward_type = self.config.trainer.ac_reward_type)

                            metrics.update(losses)

                            wm_batch.non_tensor_batch['uid'] = actor_batch.non_tensor_batch['uid']
                            wm_batch.batch['token_level_scores'] = reward_tensor
                            wm_batch.batch['token_level_rewards'] = wm_batch.batch['token_level_scores']
                            
                            wm_batch = compute_advantage(wm_batch,
                                                    adv_estimator=self.config.algorithm.adv_estimator,
                                                    gamma=self.config.algorithm.gamma,
                                                    lam=self.config.algorithm.lam,
                                                    #   num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                    uniform_std=self.config.algorithm.uniform_std)
                            
                    else:            
                        with _timer('process', timing_raw):
                            wm_batch = self.tokenizer_wg.process(wm_batch)  # "pixels" "predicted_actions" to ['input_ids', 'attention_mask', 'position_ids', 'action_ids', 'labels']
                        # breakpoint()

                        gt_seq = DataProto.from_single_dict({
                            'gt_seq': wm_batch.batch['input_ids']
                        })
                        processed_pixels = wm_batch.pop(batch_keys=['pixels'])

                        wm_batch = DataProto.from_single_dict({
                            k: v[:, :self.config.processor.gen_input_length] for k, v in wm_batch.batch.items()
                        })
                        wm_batch = wm_batch.union(gt_seq)
                        wm_batch.non_tensor_batch['uid'] = actor_batch.non_tensor_batch['uid']

                        if 'ctx_tokens' in wm_batch.batch.keys():
                            ctx_tokens = wm_batch.pop(batch_keys=['ctx_tokens'])
                        else:
                            ctx_tokens = None
                        # pop those keys for generation
                        if self.config.world_model_rollout.rollout.interact:
                            if self.config.world_model_rollout.rollout.w_gt_ac:
                                wm_gen_batch = wm_batch.pop(batch_keys=['input_ids', 'action_ids',
                                                                    'attention_mask', 'position_ids', 'gt_action_ids']
                                                        )
                            else:
                                wm_gen_batch = wm_batch.pop(batch_keys=['input_ids', 'action_ids',
                                                                    'attention_mask', 'position_ids']
                                                    )
                        else:
                            raise NotImplementedError('Currently, only support interact mode for world model rollout')
                            gen_batch = wm_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                                                # non_tensor_batch_keys=['raw_prompt_ids'],
                                                )
                        # breakpoint()
                        with _timer('wm_rollout', timing_raw):
                            wm_gen_batch_output = self.wm_rollout_wg.generate_sequences(wm_gen_batch)  # TODO: add world model function generate_seq 
                            wm_batch = wm_batch.union(wm_gen_batch_output)
                            # breakpoint()
                            if ctx_tokens is not None:
                                wm_batch = wm_batch.union(ctx_tokens)

                        # compute values
                        if self.use_critic:
                            with _timer('values', timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with _timer('adv', timing_raw):
                        # we combine with rule-based rm
                            # compute scores. Support both model and function-based.
                            # We first compute the scores using reward model. Then, we call reward_fn to combine
                            # the results from reward model and rule-based results.
                            if self.use_rm:
                                # we first compute reward model score
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            # we combine with rule-based rm
                            reward_fn = self.msp_reward_fn if self.config.processor.processor_type == 'ctx_msp' else self.reward_fn
                            # breakpoint()

                            reward_tensor, losses = reward_fn(
                                wm_batch,
                                processed_pixels,
                                return_reward_tensor=True,
                                save_pred=(self.global_steps % self.config.trainer.save_img_freq == 0),
                            )
                            
                            metrics.update(losses)
                            reward_extra_infos_dict = {}

                            wm_batch.batch['token_level_scores'] = reward_tensor
                            

                            # print(f'{list(reward_extra_infos_dict.keys())=}')
                            if reward_extra_infos_dict:
                                wm_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # breakpoint()
                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                wm_batch, kl_metrics = apply_kl_penalty(batch,
                                                                    kl_ctrl=self.kl_ctrl_in_reward,
                                                                    kl_penalty=self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                wm_batch.batch['token_level_rewards'] = wm_batch.batch['token_level_scores']

                            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                                raise NotImplementedError('REMAX is not supported yet')
                            
                            # compute advantages, executed on the driver process
                            wm_batch = compute_advantage(wm_batch,
                                                    adv_estimator=self.config.algorithm.adv_estimator,
                                                    gamma=self.config.algorithm.gamma,
                                                    lam=self.config.algorithm.lam,
                                                    #   num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                    uniform_std=self.config.algorithm.uniform_std)

                    actor_batch = actor_batch.union(wm_batch.select(batch_keys=['advantages', 'returns', 'token_level_rewards']))
                    # breakpoint()
                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                    
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(actor_batch)  # TODO: input 
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    
                    # breakpoint()
                    # validate
                    # if self.config.trainer.test_freq > 0 and \
                    #     (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                    #     with _timer('testing', timing_raw):
                    #         val_metrics: dict = self._validate()
                    #         if is_last_step:
                    #             last_val_metrics = val_metrics
                    #     metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint(save_dataloader=False)
                    elif (self.total_training_steps - self.global_steps) <= self.config.trainer.save_last_freq * self.config.trainer.save_last_num and \
                        (self.total_training_steps - self.global_steps) % self.config.trainer.save_last_freq == 0:
                        with _timer('save_last_checkpoint', timing_raw):
                            self._save_checkpoint(save_dataloader=False)

                # collect metrics
                metrics.update(compute_data_metrics(batch=actor_batch, use_critic=self.use_critic))
                # metrics.update(compute_timing_metrics(batch=actor_batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                # n_gpus = self.resource_pool_manager.get_n_gpus()
                # metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                # pickle.dump(batch, open(f"plots/{self.config.trainer.experiment_name}-batch-{self.global_steps}.pkl", "wb"))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
