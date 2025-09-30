"""
Loads a checkpoint that only has a LoRA adapter (no merged model) and merges the adapter
into the base OpenVLA model. Saves the final checkpoint in the same directory.

Make sure to specify the correct base checkpoint when running this script. For example,
- if you fine-tuned the default OpenVLA-7B model without modifications, then `--base_checkpoint=="openvla/openvla-7b"`
- if you fine-tuned a different model or resumed fine-tuning from a different checkpoint, then specify that base checkpoint
- if you fine-tuned the default OpenVLA-7B model with modifications to `modeling_prismatic.py` (OpenVLA class definition),
  then the base checkpoint path should point to the checkpoint containing the modifications

Usage:
    python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint openvla/openvla-7b \
        --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load, load_vla
import glob



@dataclass
class ConvertConfig:
    # fmt: off

    base_checkpoint: Union[str, Path] = ""                   # Base model checkpoint path/dir (either openvla/openvla-7b or whichever model you fine-tuned / resumed training from)
    lora_finetuned_checkpoint_dir: Union[str, Path] = "checkpoints/libero/rlvr/8.22/ckpt_150000_test_save/global_step_1/actor"     # Checkpoint directory containing the LoRA adapter
    vlm_path: Union[str, Path] = "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b" 
    use_minivla: bool = True                        # 


    # fmt: on


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if cfg.use_minivla:

        action_query_pattern = os.path.join(cfg.lora_finetuned_checkpoint_dir, f"action_query--*.pt")
        action_query_files = glob.glob(action_query_pattern)
        
        if action_query_files:
            action_query_path = action_query_files[0]
            print(f"Found action_query file: {action_query_path}")

        aq_state_dict = torch.load(action_query_path, weights_only=True)


        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN', '')
        vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        config = AutoConfig.from_pretrained("pretrained_models/minivla/config.json")
        config.attn_implementation='flash_attention_2'
        vla = AutoModelForVision2Seq.from_config(config).to(torch.bfloat16)
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
        ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
        # import pdb; pdb.set_trace()
        RAW_STATE_DICT["action_queries.weight"] = aq_state_dict
    
        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
    else:
        # Load Model using HF AutoClasses
        print(f"Loading base model: {cfg.base_checkpoint}")
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.base_checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    # Load LoRA weights and merge into base model, then save final checkpoint
    print("Merging LoRA weights into base model...")
    start_time = time.time()
    merged_vla = PeftModel.from_pretrained(vla, os.path.join(cfg.lora_finetuned_checkpoint_dir, "lora_adapter")).to(
        "cuda"
    )
    merged_vla = merged_vla.merge_and_unload()
    merged_vla.save_pretrained(cfg.lora_finetuned_checkpoint_dir)
    print(f"\nMerging complete! Time elapsed (sec): {time.time() - start_time}")
    print(f"\nSaved merged model checkpoint at:\n{cfg.lora_finetuned_checkpoint_dir}")


if __name__ == "__main__":
    main()
