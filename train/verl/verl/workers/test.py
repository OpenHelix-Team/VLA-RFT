
import logging
import os
import warnings
import psutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type


import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead_V1
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models import load, load_vla

from codetiming import Timer

ckpt_path='/202431205128/baseline/minivla-oft/openvla-oft/outputs/8.10/minivla+libero_4_task_suites_no_noops+b16+lr-0.0001+lora-r64+dropout-0.0--image_aug--v1--minivla--lora64a128--token_64--4_task--2025-08-10_11-15-13--50000_chkpt'
cfg_path='/202431205128/baseline/minivla-oft/pretrained_models/minivla/config.json'
fsdp_config = {
    "wrap_policy": {
        # "transformer_layer_cls_to_wrap": None,
        "min_num_params": 0
    },
    "param_offload": False,
    "optimizer_offload": False,
    "fsdp_size": -1
}
optim_config = {
    "lr": 1e-6,
    "lr_warmup_steps": -1,
    "lr_warmup_steps_ratio": 0.0,
    "min_lr_ratio": None,
    "warmup_style": "constant",
    "total_training_steps": -1,
    "weight_decay": 0.01,
    "lora_rank": 64,
    "lora_dropout": 0.0
}
num_images_in_input = 1
enable_gradient_checkpointing = False
trust_remote_code = False
use_liger = False
role = 'actor'
from verl.utils.model import print_model_size
from verl.utils.torch_dtypes import PrecisionType
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForVision2Seq, AutoImageProcessor, AutoProcessor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
from torch import optim
from experiments.robot.openvla_utils import update_auto_map, check_model_logic_mismatch, _load_dataset_stats, find_checkpoint_file, load_component_state_dict
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

import torch.distributed
# # 修复分布式初始化问题 - 设置单机环境变量
# if not torch.distributed.is_initialized():
#     # 设置单机分布式环境变量
#     os.environ.setdefault('RANK', '0')
#     os.environ.setdefault('WORLD_SIZE', '1')
#     os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
#     os.environ.setdefault('MASTER_PORT', '12355')
    
#     # 检查CUDA是否可用
#     if torch.cuda.is_available():
#         backend = 'nccl'
#     else:
#         backend = 'gloo'
        
#     try:
#         torch.distributed.init_process_group(backend=backend, rank=0, world_size=1)
#         print(f"Initialized distributed with backend: {backend}")
#     except Exception as e:
#         print(f"Failed to initialize distributed: {e}")
#         print("Continuing without distributed training...")

def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        device_mesh = init_device_mesh('cuda',
                                       mesh_shape=(world_size // fsdp_size, fsdp_size),
                                       mesh_dim_names=['ddp', 'fsdp'])
    return device_mesh

# build device mesh for FSDP
if torch.distributed.is_initialized():
    world_size = torch.distributed.get_world_size()
else:
    world_size = 2  # Default to 2 for single-node testing

print(f"World size: {world_size}")
# TODO(sgm): support FSDP hybrid shard for larger model
# 修复字典访问问题
device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_config['fsdp_size'])

print(f"World size: {world_size}")
print(f"Device mesh: {device_mesh}")

update_auto_map(ckpt_path)
check_model_logic_mismatch(ckpt_path)

torch_dtype = fsdp_config.get('model_dtype', None)
if torch_dtype is None:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = PrecisionType.to_dtype(torch_dtype)

# override model kwargs
actor_model_config = AutoConfig.from_pretrained(cfg_path, trust_remote_code=trust_remote_code)
actor_model_config.attn_implementation='flash_attention_2'

print(f'Model config after override: {actor_model_config}')

# NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings,
                                                mesh=device_mesh)

with init_context(), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
        actor_module_class = AutoModelForVision2Seq
    else:
        raise ValueError(f'{type(actor_model_config)} is not supported')
        actor_module_class = AutoModelForCausalLM
    
    actor_module = actor_module_class.from_pretrained(pretrained_model_name_or_path=ckpt_path,
                                                        torch_dtype=torch_dtype,
                                                        attn_implementation='flash_attention_2',
                                                        low_cpu_mem_usage=False,
                                                        trust_remote_code=trust_remote_code)

    actor_module.vision_backbone.set_num_images_in_input(num_images_in_input)
    _load_dataset_stats(actor_module, ckpt_path)

    # Apply Liger kernel to the model if use_liger is set to True
    if use_liger:
        from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
        _apply_liger_kernel_to_instance(model=actor_module)
    
    # 修复 optim_config 访问问题
    if optim_config is not None and optim_config.get('lora_rank', 0) > 0:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=optim_config['lora_rank'],
            lora_alpha=2 * optim_config['lora_rank'],
            lora_dropout=optim_config.get('lora_dropout', 0.0),
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        actor_module = get_peft_model(actor_module, lora_config)
        for name, param in actor_module.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        actor_module.print_trainable_parameters()

    actor_module.to(torch_dtype)

    if enable_gradient_checkpointing:
        actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

if torch.distributed.is_initialized():
    torch.distributed.barrier()

actor_auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None), is_lora=optim_config.get('lora_rank', 0) > 0)

print(f'actor_wrap_policy: {actor_auto_wrap_policy}')

def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy

fsdp_mesh = device_mesh
sharding_strategy = get_sharding_strategy(fsdp_mesh)

ACTION_DIM = 7
PROPRIO_DIM = 8
NUM_DIFFUSION_STEPS = 50
from torch.nn.parallel import DistributedDataParallel as DDP
# note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
# TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=trust_remote_code)
device = torch.cuda.current_device()
llm_dim = actor_module.llm_dim
print(f'actor_module.llm_dim_origin: {actor_module.llm_dim}')
proprio_projector = ProprioProjector(
    llm_dim=llm_dim,
    proprio_dim=PROPRIO_DIM,
    ).to(device=device, dtype=torch.bfloat16)
proprio_projector_path = find_checkpoint_file(ckpt_path, "proprio_projector")
proprio_state_dict = load_component_state_dict(proprio_projector_path)
proprio_projector.load_state_dict(proprio_state_dict)
proprio_projector = DDP(proprio_projector, device_ids=[device], gradient_as_bucket_view=True, device_mesh=fsdp_mesh)

action_head = DiffusionActionHead_V1(
        input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, num_diffusion_steps=NUM_DIFFUSION_STEPS
    ).to(device=device, dtype=torch.bfloat16)
action_head_path = find_checkpoint_file(ckpt_path, "action_head")
action_head_state_dict = load_component_state_dict(action_head_path)
action_head.load_state_dict(action_head_state_dict)

noisy_action_projector = NoisyActionProjector(
    llm_dim=llm_dim).to(device=device, dtype=torch.bfloat16)
noisy_action_projector_path = find_checkpoint_file(ckpt_path, "noisy_action_projector")
noisy_action_projector_state_dict = load_component_state_dict(noisy_action_projector_path)
noisy_action_projector.load_state_dict(noisy_action_projector_state_dict)
noisy_action_projector = DDP(noisy_action_projector, device_ids=[device], gradient_as_bucket_view=True, device_mesh=fsdp_mesh)

# TODO: add transformer policy
# We force reference policy to use CPUOffload to save memory.
# We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
cpu_offload = None if role == 'actor' else CPUOffload(offload_params=True)
actor_module_fsdp = FSDP(
    actor_module,
    cpu_offload=cpu_offload,
    param_init_fn=init_fn,
    use_orig_params=True,
    auto_wrap_policy=actor_auto_wrap_policy,
    device_id=torch.cuda.current_device(),
    sharding_strategy=sharding_strategy,  # zero3
    sync_module_states=True,
    device_mesh=device_mesh,
    forward_prefetch=False)

action_head_fsdp = FSDP(
    action_head,
    cpu_offload=cpu_offload,
    param_init_fn=init_fn,
    use_orig_params=True,
    device_id=torch.cuda.current_device(),
    sharding_strategy=sharding_strategy,  # zero3
    sync_module_states=True,
    device_mesh=device_mesh,
    forward_prefetch=False)
action_head = action_head_fsdp._fsdp_wrapped_module

# TODO: add more optimizer args into config
if role == 'actor' and optim_config is not None:
    from verl.utils.torch_functional import get_constant_schedule_with_warmup
    trainable_params = [param for param in actor_module_fsdp.parameters() if param.requires_grad]
    trainable_params += [param for param in action_head_fsdp.parameters() if param.requires_grad]
    trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    
    # 修复字典访问
    actor_optimizer = optim.AdamW(trainable_params,
                                    lr=optim_config['lr'],
                                    betas=optim_config.get('betas', (0.9, 0.999)),
                                    weight_decay=optim_config.get('weight_decay', 1e-2))

    total_steps = optim_config.get('total_training_steps', 0)
    num_warmup_steps = int(optim_config.get('lr_warmup_steps', -1))
    if num_warmup_steps < 0:
        num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

    print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

    actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                            num_warmup_steps=num_warmup_steps)
else:
    actor_optimizer = None
    actor_lr_scheduler = None

actor_module = actor_module_fsdp._fsdp_wrapped_module

actor_module_fsdp.eval()
action_head_fsdp.eval()
proprio_projector.eval()
noisy_action_projector.eval()


# print(f"actor_module_fsdp: {actor_module_fsdp}, actor_module_optimizer: {actor_optimizer}, \
#       actor_lr_scheduler: {actor_lr_scheduler}, actor_module_config: {actor_model_config}")
print("Model and optimizer setup completed successfully!")

from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform_V1
from torch.utils.data import DataLoader
action_tokenizer = ActionTokenizer(processor.tokenizer)
batch_transform = RLDSBatchTransform_V1(
    action_tokenizer,
    processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder,
    use_wrist_image=False,
    use_proprio=True,
    use_minivla=True,
    use_raw_image=True
)
train_dataset = RLDSDataset(
    '/202431205128/baseline/minivla-oft/data/modified_libero_rlds',
    'libero_4_task_suites_no_noops',
    batch_transform,
    resize_resolution=tuple(actor_module_fsdp._fsdp_wrapped_module.config.image_sizes),
    shuffle_buffer_size=100_000,
    image_aug=True,
)
breakpoint()
collator = PaddedCollatorForActionPrediction(
    processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
)
dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    sampler=None,
    collate_fn=collator,
    num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
print('all done!!!!!!!!!!')

import tqdm

with tqdm.tqdm(total=10, leave=False) as progress:
    for batch_idx, batch in enumerate(dataloader):
        breakpoint()
        break