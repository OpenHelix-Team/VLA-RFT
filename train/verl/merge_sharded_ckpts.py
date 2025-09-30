import os
import torch
import glob
from safetensors.torch import save_file
from torch.distributed._tensor import DTensor
import torch.distributed as dist
from collections import OrderedDict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', required=True)
parser.add_argument('--config_path', default='checkpoints/worldmodel/final_version/backbone')
args = parser.parse_args()

def dtensor_to_tensor(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, DTensor):
            # Extract local tensor part from DTensor
            new_value = value._local_tensor.detach().clone()
        else:
            new_value = value.detach().clone() if torch.is_tensor(value) else value
        new_state_dict[key] = new_value
    return new_state_dict


def merge_fsdp_shards(checkpoint_dir, world_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    full_state_dict = OrderedDict()
    shard_metadata = {}  # Record sharding information for each parameter

    # First pass: collect parameter metadata
    for rank in range(world_size):
        chunk = torch.load(f"{checkpoint_dir}/model_world_size_{world_size}_rank_{rank}.pt",
                           map_location='cpu')
        for key, value in chunk.items():
            if key not in shard_metadata:
                shard_metadata[key] = {
                    'shape': value.shape,
                    'dtype': value.dtype,
                    # 'is_sharded': len(value.shape) > 0 and value.shape[0] % world_size == 0
                    'is_sharded': len(value.shape) > 0
                }

    # Second pass: merge parameters
    for key in shard_metadata:
        if shard_metadata[key]['is_sharded']:
            # Parameters that need to be concatenated
            shards = []
            for rank in range(world_size):
                chunk = torch.load(f"{checkpoint_dir}/model_world_size_{world_size}_rank_{rank}.pt",
                                   map_location='cpu')
                if isinstance(chunk[key], DTensor):
                    # 提取 DTensor 的本地张量部分
                    new_value = chunk[key]._local_tensor.detach().clone()
                else:
                    new_value = chunk[key].detach().clone() if torch.is_tensor(chunk[key]) else chunk[key]
                shards.append(new_value)
            full_state_dict[key] = torch.cat(shards, dim=0)
        else:
            # 不需要拼接的完整参数（取rank 0的副本）
            chunk = torch.load(f"{checkpoint_dir}/model_world_size_{world_size}_rank_0.pt",
                               map_location='cpu')
            full_state_dict[key] = chunk[key]

    # 处理FSDP前缀
    full_state_dict = {k.replace("_fsdp_wrapped_module.", ""): v
                       for k, v in full_state_dict.items()}

    save_file(full_state_dict, os.path.join(output_dir, 'model.safetensors'))
    # torch.save(full_state_dict, os.path.join(output_dir, 'model.pt'))

output_dir = os.path.join(args.ckpt_path, 'merged_ckpt')
merge_fsdp_shards(args.ckpt_path, 4, output_dir)

jsons = glob.glob(os.path.join(args.config_path, '*.json'))
for json in jsons:
    cmd = f"cp {json} {output_dir}"
    os.system(cmd)
