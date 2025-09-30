"""
deploy.py

Starts VLA server which the client can query to get robot actions.
"""

import os.path
import pickle

# ruff: noqa: E402
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch


from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_noisy_action_projector,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
)
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenHelixServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.cfg = cfg

        # Load model
        self.vla = get_vla(cfg)

        # Load proprio projector
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

        # Load continuous action head
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        # Check that the model contains the action un-normalization key

        assert cfg.unnorm_key in self.vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # Get Hugging Face processor
        self.processor = None
        self.processor = get_processor(cfg)

        self.noisy_action_projector = None

        if cfg.use_diffusion:
            self.noisy_action_projector = get_noisy_action_projector(cfg, self.vla.llm_dim)

    

    def get_action(self, obs: Dict[str, Any], instru: str) -> str:
        # obs["full_image"]: static image
        # obs["wrist"]: wrist image
        # obs["state"]: 6-dim proprio state 

        # instru: task description

        try:
            observation = obs
            instruction = instru

            action = get_vla_action(
                cfg=self.cfg, 
                vla=self.vla, 
                processor=self.processor, 
                obs=observation, 
                task_label=instruction, 
                action_head=self.action_head, 
                proprio_projector=self.proprio_projector, 
                use_film=self.cfg.use_film,
                noisy_action_projector=self.noisy_action_projector
            )
            return action
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "obs: Dict[str, Any], instru: str \n"
            )
            return "error"



@dataclass
class DeployConfig:
    # fmt: off

    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""    # Pretrained checkpoint path
    use_minivla: bool = True                   # If True, 

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_flow_matching: bool = False
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""     # realworld? # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "v1"                        # version of exps


if __name__ == "__main__":
    cfg = DeployConfig()
    cfg.pretrained_checkpoint = "/ssdwork/Pengxiang/code/minivla-oft/openvla-oft/outputs/minivla+blue_block+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--realworld--miniv1--parallel_dec--8_acts_chunk--continuous_acts--L1--lora_r64--lr2e-4--2025-07-19_06-31-22--100_chkpt" 
    cfg.unnorm_key = "blue_block" 
    cfg.save_version = "v1" 
    server = OpenHelixServer(cfg)
    with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
        observation = pickle.load(file)
    action = server.get_action(obs={"full_image": observation["full_image"], "wrist": observation["wrist_image"], "state": observation["state"][:6]}, instru=observation["task_description"])
    print(action)