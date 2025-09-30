
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import random
import torch.nn as nn
import numpy as np
import transformers
from timm.models.vision_transformer import Mlp, PatchEmbed
from prismatic.models.transformer_utils import CrossAttentionBlock, PerceiverResampler, MAPBlock

torch.set_printoptions(threshold=np.inf)

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, RoPE=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.RoPE = False
        # if RoPE:
        #     from rotary_embedding_torch import RotaryEmbedding
        #     self.rotary_emb = RotaryEmbedding(dim = head_dim // 2)

    def forward(self, x, mask = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        if self.RoPE:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            x = xformers.ops.memory_efficient_attention(q, k, v).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale

            if mask is not None:
                _MASKING_VALUE = -1e+30 if attn.dtype == torch.float32 else -1e+4
                attn.masked_fill(mask.to(attn.device) == 0, _MASKING_VALUE)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)


        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_bfp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_bfp16:
            t_freq = t_freq.to(dtype=torch.bfloat16)
        t_emb = self.mlp(t_freq)
        return t_emb



#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiffusionTransformerBlock(nn.Module):
    """
    A DiT tansformer block with adaptive layer norm zero (adaLN-Zero) and cross-attention conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_temporal = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.cross_attn = CrossAttentionBlock(  v_dim=hidden_size,
                                                l_dim=hidden_size,             
                                                embed_dim=hidden_size,
                                                num_heads=num_heads,)

    def forward(self, x, c, context_embed, n_batches=1, attn_mask = None):
        # add-LN conditioning
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Spatial(action-wise) attention
        x = x + gate_msa.unsqueeze(1) * self.attn_temporal(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)

        # Cross-attn conditioning
        # import pdb; pdb.set_trace()
        x = self.cross_attn(x, context_embed)

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# Core Specialist Policy Implementation
class DiT_SingleTokenAction(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels,
        out_channels=7,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_actions=8,
        attention_mode='math',
    ):
        super().__init__()
        self.out_channels =  out_channels
        self.num_heads = num_heads
        self.num_actions = num_actions

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.proprio_embedder = nn.Linear(896, hidden_size)

        self.context_adapter = nn.Linear(896, hidden_size)

        self.temp_embed = nn.Parameter(torch.zeros(1, self.num_actions, hidden_size), requires_grad=False)                
        self.hidden_size = hidden_size

        temporal_len = self.num_actions
        # self.causal_mask = torch.tril(torch.ones(temporal_len, temporal_len)).view(1,1,temporal_len,temporal_len)
        self.full_mask = torch.ones(temporal_len, temporal_len).view(1,1,temporal_len,temporal_len)

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode, attn_drop=0.1, RoPE=True) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels)
        self.initialize_weights()



    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.proprio_embedder.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        # nn.init.xavier_uniform_(self.final_layer.linear.weight)
        # nn.init.zeros_(self.final_layer.linear.bias)



    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self, 
                x, 
                timesteps, 
                context=None,
                proprio=None, 
                use_fp16=False):
        """
        Forward pass of DiT.
        x: (N, T, d_action) tensor of actions
        t: (N,) tensor of diffusion timesteps
        cond: (N, T, d_cond) tensor of conditions
        """
        # import pdb;pdb.set_trace()
        # if len(timesteps.shape) == 0:
        #     timesteps = timesteps[None]
        # timesteps = .to(x.device)

        # if use_fp16:
        #     x = x.to(dtype=torch.float16)
        batches, f, d_action = x.shape

        x = self.x_embedder(x) + self.temp_embed
        t = self.t_embedder(timesteps, use_bfp16=True)
        # t = self.t_embedder(timesteps)
        # t = timesteps

        proprio = self.proprio_embedder(proprio)
        # import pdb; pdb.set_trace()
        # print(context.size())
        context = self.context_adapter(context)

        # import pdb; pdb.set_trace()
        # print(context.size())
        context_embed = context
        global_cond = proprio + t
        # global_cond = global_cond + 
        # global_cond = global_cond.squeeze(1)

       
        for i in range(0, len(self.blocks)):
            x = self.blocks[i]( 
                            x = x, 
                            c = (global_cond + context[:,i+1,:,:].mean(dim=1, keepdims=True)).squeeze(1), 
                            n_batches = batches, 
                            context_embed = context_embed[:,i+1,:,:], 
                            attn_mask = self.full_mask, 
                )
        # import pdb; pdb.set_trace()
        # print(context.size())
        x = self.final_layer(x, (global_cond+context[:,-1,:,:].mean(dim=1, keepdims=True)).squeeze(1))    
        # print(context.size())                         
        return x


class DiT_SingleTokenAction_OneCtx(DiT_SingleTokenAction):
    """
    A DiT variant that supports *single-layer* context.

    Acceptable context input shapes (before linear projection):
      - (B, S_ctx, 896)
      - (B, 1, S_ctx, 896)
      - (B, Lp1, S_ctx, 896)  # still works; will be padded/expanded as needed

    Internally the single context layer is broadcast across all transformer blocks
    (depth) and the final layer (hence depth+1 entries are required).
    """
    def __init__(
        self,
        in_channels,
        out_channels=7,
        hidden_size=1152,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        num_actions=8,
        attention_mode='math',
        ctx_every=2,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_actions=num_actions,
            attention_mode=attention_mode,
        )
        self.ctx_every = ctx_every

    def _prepare_context(self, context: torch.Tensor) -> torch.Tensor:
        """
        Normalize context tensor to shape (B, depth+1, S_ctx, hidden_size) after projection.
        Input accepted shapes (last dim must be 896 before projection):
            (B, S_ctx, 896)            -> unsqueeze to (B, 1, S_ctx, 896)
            (B, 1, S_ctx, 896)        -> keep
            (B, Lp1, S_ctx, 896)      -> keep (Lp1 may be <, =, or > depth+1)
        Then apply self.context_adapter to reach hidden_size and broadcast/pad to (B, depth+1, S_ctx, hidden_size).
        """
        if context is None:
            raise ValueError("`context` cannot be None for DiT_SingleTokenAction_OneCtx.")

        # Ensure 4D: (B, Lp1?, S_ctx, 896)
        if context.dim() == 3:
            # (B, S_ctx, 896) -> (B, 1, S_ctx, 896)
            context = context.unsqueeze(1)
        elif context.dim() != 4:
            raise ValueError(
                f"Unsupported context.dim()={context.dim()}. Expected 3 or 4."
            )

        B, Lp1, S_ctx, Dctx = context.shape
        if Dctx != 896:
            raise ValueError(
                f"Expected context last dim = 896 before projection, got {Dctx}."
            )

        # Project to hidden size; Linear supports N-D input (applies on last dim)
        context = self.context_adapter(context)  # (B, Lp1, S_ctx, hidden)

        # We need exactly (depth + 1) context slices: one per block, plus one for final layer
        need = len(self.blocks) + 1
        if Lp1 == 1:
            # Broadcast the single layer across all required slices
            context = context.expand(-1, need, -1, -1).contiguous()
        elif Lp1 < need:
            # Pad by repeating the last slice
            pad_reps = need - Lp1
            last = context[:, -1:, :, :].expand(-1, pad_reps, -1, -1)
            context = torch.cat([context, last], dim=1)
        elif Lp1 > need:
            # Truncate to the needed length (keep the first `need` slices)
            context = context[:, :need, :, :]

        return context  # (B, depth+1, S_ctx, hidden)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor = None,
        proprio: torch.Tensor = None,
        use_fp16: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with single-layer (or broadcastable) context.

        Args:
            x:          (B, T_actions, d_action)
            timesteps:  (B,)
            context:    (B, S_ctx, 896) or (B, 1, S_ctx, 896) or (B, Lp1, S_ctx, 896)
            proprio:    (B, 896)
        Returns:
            (B, T_actions, out_channels)
        """
        # Shapes
        B, T_actions, d_action = x.shape

        # Embeddings
        x = self.x_embedder(x) + self.temp_embed  # (B, T_actions, hidden)

        # Timestep & proprio embeddings
        t_emb = self.t_embedder(timesteps, use_bfp16=True)            # (B, hidden)
        proprio_emb = self.proprio_embedder(proprio)                  # (B, hidden)
        global_cond = proprio_emb + t_emb                             # (B, hidden)

        # Prepare context to (B, depth+1, S_ctx, hidden)
        context = self._prepare_context(context)
        context_embed = context                                        # (B, depth+1, S_ctx, hidden)

        # Attention mask (1,1,T,T)
        attn_mask = self.full_mask.to(x.device)

        # Transformer blocks with sparse cross-attn (ctx_every)
        for i, block in enumerate(self.blocks):
            # Per-layer conditioning vector by averaging tokens in its context slice
            c_i = (global_cond + context[:, i, :, :].mean(dim=1, keepdims=True)).squeeze(1)  # (B, hidden)

            use_cross = (i % self.ctx_every == 0) or (i == len(self.blocks) - 1) or (i == 0)
            if use_cross:
                x = block(
                    x=x,
                    c=c_i,
                    n_batches=B,
                    context_embed=context_embed[:, i, :, :],
                    attn_mask=attn_mask,
                )
            else:
                # Skip cross-attn: run MSA + MLP with adaLN-Zero
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN_modulation(c_i).chunk(6, dim=1)
                x = x + gate_msa.unsqueeze(1) * block.attn_temporal(
                    modulate(block.norm1(x), shift_msa, scale_msa), attn_mask
                )
                x = x + gate_mlp.unsqueeze(1) * block.mlp(
                    modulate(block.norm3(x), shift_mlp, scale_mlp)
                )

        # Final layer uses the last context slice
        final_c = (global_cond + context[:, -1, :, :].mean(dim=1, keepdims=True)).squeeze(1)  # (B, hidden)
        x = self.final_layer(x, final_c)  # (B, T_actions, out_channels)
        return x



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################
def DiT_Large_STA(**kwargs):
    return DiT_SingleTokenAction(depth=18, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def DiT_Base_STA(**kwargs):
    return DiT_SingleTokenAction(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def DiT_Small_STA(**kwargs):
    return DiT_SingleTokenAction(depth=8, hidden_size=256, patch_size=1, num_heads=8, **kwargs)

def DiT_Tiny_STA(**kwargs):
    return DiT_SingleTokenAction(depth=6, hidden_size=256, patch_size=1, num_heads=8, **kwargs)

# Convenience factory with depth=8
def DiT_Depth8_STA_OneCtx(**kwargs):
    kwargs.setdefault("depth", 8)
    kwargs.setdefault("ctx_every", 2)
    return DiT_SingleTokenAction_OneCtx(**kwargs)
