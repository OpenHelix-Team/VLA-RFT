from typing import *

import torch
import torch.nn as nn
import numpy as np

from dataclasses import dataclass
from diffusers.models.autoencoders.vae import VectorQuantizer
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook

from .vae import CNNEncoder, CNNDecoder
from .finite_scalar_quantize import FSQ, get_fsq_levels


@dataclass
class CompressiveVQDecoderOutput(BaseOutput):

    sample: torch.FloatTensor
    commit_loss: Optional[torch.FloatTensor] = None


class CNNFSQModel256(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D", "DownEncoderBlock2D",
                                             "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D", "UpDecoderBlock2D",
                                           "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512, 768),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 64,
        norm_num_groups: int = 32,
        fsq_levels=12,
        norm_type: str = "group",  # group, spatial
        resolution=256,
    ):
        super().__init__()
        if isinstance(fsq_levels, int):
            fsq_levels = get_fsq_levels(fsq_levels)

        self.latent_channels = latent_channels

        # encoders
        self.encoder = CNNEncoder(
            in_channels=in_channels,
            out_channels=self.latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
            mid_block_add_attention=True,
            init_res=resolution,
        )

        # vector quantization
        self.fsq_levels = fsq_levels
        self.vq_embed_dim = len(fsq_levels)
        self.num_vq_embeddings = np.prod(fsq_levels)

        self.quant_linear = nn.Conv2d(self.latent_channels, self.vq_embed_dim, kernel_size=1, stride=1, padding=0)
        self.dynamics_quantize = FSQ(
            levels=fsq_levels,
        )
        self.post_quant_linear = nn.Conv2d(self.vq_embed_dim, self.latent_channels, kernel_size=1, stride=1, padding=0)

        # decoders
        self.decoder = CNNDecoder(
            in_channels=self.latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_type=norm_type,
            mid_block_add_attention=True,
            init_res=resolution // 2**(len(down_block_types) - 1),
        )

    def codes_to_indices(self, x):
        return self.dynamics_quantize.codes_to_indices(x)

    def indices_to_codes(self, x):
        return self.dynamics_quantize.indices_to_codes(x)

    def forward(
        self,
        sample: torch.FloatTensor,
    ) -> CompressiveVQDecoderOutput:
        """
        sample: [B*T, C, H, W] or [B, T, C, H, W]
        cond_features: [B*T, C, H, W] or [B, T, C, H, W]
        """
        if sample.ndim == 5:
            input = sample.reshape(-1, *sample.shape[2:])
        else:
            input = sample

        # encode
        d = self.encoder(input)

        # go through quantization layer
        d = self.quant_linear(d)
        quant, info = self.dynamics_quantize(d)
        commit_loss = torch.tensor(0.0).to(sample.device)
        quant2 = self.post_quant_linear(quant)

        # decode
        dec = self.decoder(quant2)

        self.code_util = torch.unique(info).shape[0] / self.num_vq_embeddings
        self.unique_code = torch.unique(info)

        dec = dec.reshape(*sample.shape)
        return dec, commit_loss

    def encode(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        if sample.ndim == 5:
            input = sample.reshape(-1, *sample.shape[2:])
        else:
            input = sample
        d = self.encoder(input)
        d = self.quant_linear(d)
        quant, info = self.dynamics_quantize(d)
        info = info.reshape(*sample.shape[:2], *info.shape[-2:])
        return info

    def decode(self, indices: torch.FloatTensor) -> torch.FloatTensor:
        if indices.ndim == 4:
            input = indices.reshape(-1, *indices.shape[2:])
        else:
            input = indices
        code = self.dynamics_quantize.indices_to_codes(input)
        quant2 = self.post_quant_linear(code)
        dec = self.decoder(quant2)
        dec = dec.reshape(*indices.shape[:-2], *dec.shape[-3:])
        return dec
