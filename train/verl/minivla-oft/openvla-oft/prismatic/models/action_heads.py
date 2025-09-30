"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS
from prismatic.models.diffusion_transformer import DiT_SingleTokenAction, DiT_SingleTokenAction_OneCtx



class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=4, input_dim=input_dim*ACTION_DIM, hidden_dim=4096, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps=100,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps, beta_schedule="squaredcos_cap_v2")
        self.num_diffusion_steps = num_diffusion_steps
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred

class CrossAttention(nn.Module):
    """Cross-Attention module to process conditional inputs."""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, cond):
        """
        x: (batch_size, seq_len, hidden_dim)
        cond: (batch_size, cond_len, hidden_dim)
        """
        # Compute Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(cond)  # (batch_size, cond_len, hidden_dim)
        V = self.value(cond)  # (batch_size, cond_len, hidden_dim)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # (batch_size, seq_len, cond_len)
        attn_scores = self.softmax(attn_scores / (K.size(-1) ** 0.5))  # Scaled dot-product attention

        # Compute attended values
        attended_values = torch.matmul(attn_scores, V)  # (batch_size, seq_len, hidden_dim)
        return attended_values

class NoisePredictionDiT_V1(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.dit = DiT_SingleTokenAction(in_channels=transformer_hidden_dim, out_channels=action_dim, depth=24, hidden_size=hidden_dim, num_heads=8)

    def forward(
        self,
        obs, hidden_states = None, time_step=None, proprio_states = None
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        # import pdb; pdb.set_trace()
        # hidden_states.size()[16, 24, 16, 896]
        output = self.dit(x = obs, context = hidden_states, timesteps = time_step, proprio = proprio_states)
        return output

class FlowPredictionDiT_V1(nn.Module):
    """
    Diffusion flow prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a flow prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.dit = DiT_SingleTokenAction_OneCtx(in_channels=transformer_hidden_dim, out_channels=action_dim, depth=8, hidden_size=hidden_dim, num_heads=8, ctx_every=2)

    def forward(
        self,
        obs, hidden_states = None, time_step=None, proprio_states = None
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        # import pdb; pdb.set_trace()
        # hidden_states.size()[16, 24, 16, 896]
        # breakpoint()
        output = self.dit(x = obs, context = hidden_states, timesteps = time_step, proprio = proprio_states)
        return output


class MLPResNetBlock_V1(nn.Module):
    """One MLP ResNet block with a residual connection and Cross-Attention conditions."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        
        # Cross-Attention module (shared for all conditions)
        self.cross_attention = CrossAttention(dim)

        # norm 
        # self.layer_norm = nn.Linear(dim, dim)

    def forward(self, x, h=None, t=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """
        # Concatenate all conditions (h, t, p) along the sequence dimension
        conditions = []
        if h is not None:
            conditions.append(h)
        if t is not None:
            conditions.append(t)
        if p is not None:
            conditions.append(p)


        if conditions:
            # Concatenate conditions along the sequence dimension (cond_len)
            cond = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)
            # Apply Cross-Attention
            # import pdb; pdb.set_trace()
            condition_output = self.cross_attention(x, cond)  # (batch_size, seq_len, hidden_dim)
            # Add the condition output to the input
            x = x + condition_output
        # import pdb; pdb.set_trace()

        # Pass through the feedforward network
        x = self.ffn(x) + x
        
        # Add the residual connection
        return x

class Connector_Block(nn.Module):
    """One MLP ResNet block with a residual connection and Multi-Head Attention conditions."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim

        self.layer_norm1 = nn.LayerNorm(dim)

        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        # Multi-head attention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, h):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            h, t, p: (batch_size, 1, hidden_dim) or None
        Returns:
            Output tensor after applying multi-head attention and feedforward network.
        """
        residual = h

        h = self.layer_norm1(h)

        # Multi-head attention
        h, _ = self.multihead_attention(query=h, key=h, value=h)  # (batch_size, seq_len, hidden_dim)

        h = residual + h

        residual = h

        # Add residual connection and feed the output through the FFN
        h = self.ffn(h)

        h = residual + h

        return h

class Connector(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, hidden_dim):
        super().__init__()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(Connector_Block(dim=hidden_dim))

    def forward(self, x):
        for i, block in enumerate(self.mlp_resnet_blocks):
            # import pdb; pdb.set_trace()
            x = block(x)  # shape: (batch_size, hidden_dim)
        return x

class L1RegressionActionHead_V1(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = MLPResNet_V1(
            num_blocks=24, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states, proprio= None, proprio_projector=None, phase='stage1'):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        # import pdb; pdb.set_trace()

        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        # import pdb; pdb.set_trace()

        task_hidden_states = actions_hidden_states[:,:,:self.num_task_tokens,:]
        actions_hidden_states = actions_hidden_states[:,:,self.num_task_tokens:,:]

        cond_actions_hidden_states = torch.zeros((batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim), device=device, dtype=actions_hidden_states.dtype).detach()

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)

        action = self.model(rearranged_actions_hidden_states, h_a = actions_hidden_states, p = proprio_features, h_t = task_hidden_states, phase=phase)
        # import pdb; pdb.set_trace()

        return action

class L1RegressionActionHead_V1_backup(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_task_tokens = num_task_tokens
        self.model = MLPResNet_V1_1(num_blocks=24, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim)
        # self.position_embedding = SinusoidalPositionalEncoding(dim=hidden_dim)
        # self.action_head = MLPResNet(num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim)
        self.mlp_image = Connector(num_blocks=6, hidden_dim=hidden_dim)
        self.mlp_action = Connector(num_blocks=6, hidden_dim=hidden_dim)
        # self.aux_action_head = MLPResNet(num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim)


    def predict_action(self, all_hidden_states, proprio= None, proprio_projector=None, vision_backbone=None):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        # import pdb; pdb.set_trace()

        batch_size = all_hidden_states.shape[0]
        device = all_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        # import pdb; pdb.set_trace()

        task_hidden_states = all_hidden_states[:,:,:self.num_task_tokens+1,:]
        actions_hidden_states = all_hidden_states[:,:,self.num_task_tokens+1:,:]
        # actions_hidden_states_maohao = all_hidden_states[:,:,self.num_task_tokens:self.num_task_tokens+1,:]
        # actions_hidden_states = actions_hidden_states + actions_hidden_states_maohao

        task_hidden_states = task_hidden_states[:,0,:,:]
        task_hidden_states = self.mlp_image(task_hidden_states)

        # import pdb; pdb.set_trace()
        actions_hidden_states = actions_hidden_states[:,:,-1,:]
        actions_hidden_states = self.mlp_action(actions_hidden_states)

        # rearranged_aux_actions_hidden_states = actions_hidden_states_last.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)

        # aux_action = self.aux_action_head(rearranged_aux_actions_hidden_states)

        # import pdb; pdb.set_trace()
        # action = self.action_head(actions_hidden_states)

        cond_actions_hidden_states = torch.zeros((batch_size, NUM_ACTIONS_CHUNK, self.hidden_dim), device=device, dtype=task_hidden_states.dtype).detach()
        # cond_actions_hidden_states = self.position_embedding(cond_actions_hidden_states) 
        # rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)

        action = self.model(cond_actions_hidden_states, h_t = task_hidden_states, h_a = actions_hidden_states, p = proprio_features)

        # action = self.action_head(rearranged_actions_hidden_states)

        return action, None

class MLPResNet_V1(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock_V1_2(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self.layer_norm3 = nn.LayerNorm(hidden_dim)
        # self.layer_norm4 = nn.LayerNorm(hidden_dim)
        # self.layer_norm5 = nn.LayerNorm(hidden_dim)

    def forward(self, x, h_a=None, h_t=None, p= None, phase='stage1'):
 
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            # import pdb; pdb.set_trace()
            x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p, phase=phase)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x    

class MLPResNet_V1_1(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, input_dim, num_blocks, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock_V1_3(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_t = None, h_a = None, p = None):
        # x: (batch_size, input_dim)   
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            # import pdb; pdb.set_trace()
            x = block(x, h_t = h_t, h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x  

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))

    def forward(self, x, h=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """
        # import pdb; pdb.set_trace()
        h = torch.cat((h ,p),dim=1)
        # import pdb; pdb.set_trace()


        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K = h.size(1)
        g = self.gating_factor

        adapter_k = h
        adapter_v = h

        # import pdb; pdb.set_trace()
        q_1 = self.q_proj(x) # (B, T, C)
        
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) # (B, H, T, K)
        
        gated_attn_scores_adapter = attn_scores_adapter * nn.Tanh()(g)
        
        attn_scores = torch.cat([attn_scores_tokens, gated_attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)
        
        v_combined = torch.cat([v_tokens, v_adapter], dim=2) # (B, H, T+K, head_dim)
        
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)
        
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        return output

class MLPResNetBlock_V1_2(nn.Module):
    """One MLP ResNet block with a residual connection and Cross-Attention conditions."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))

        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)


    def forward(self, x, h_t=None, h_a=None, p=None, phase='stage1'):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor

        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        # import pdb; pdb.set_trace()

        q_1 = self.q_proj(x) # (B, T, C)


        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)


        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) # (B, H, T, K)

        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) # (B, H, T, K)

        gated_scores_adapter = attn_scores_adapter * ratio_g
        
        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, gated_scores_adapter], dim=-1) # (B, H, T, T+K)
        
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)
        
        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)

        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)
        
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        output = self.attention_dropout(output)

        x = self.ffn(output + x) 
        x = self.ffn_dropout(x)


        return x


class MLPResNetBlock_V1_3(nn.Module):
    """One MLP ResNet block with a residual connection and Multi-Head Attention conditions."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim

        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        # Multi-head attention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            h, t, p: (batch_size, 1, hidden_dim) or None
        Returns:
            Output tensor after applying multi-head attention and feedforward network.
        """
        # Concatenate all conditions (h, t, p) along the sequence dimension
        # import pdb; pdb.set_trace()
        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        h_a = h_a.reshape(B,-1,C)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if h_t is not None:
            conditions.append(h_t)
        if p is not None:
            conditions.append(p)

        if len(conditions) > 0:
            cond = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)
            inputs = torch.cat([x, cond], dim=1)  # (batch_size, seq_len + cond_len, hidden_dim)
        else:
            inputs = x  # No conditions, use only x

        # Multi-head attention
        output, _ = self.multihead_attention(query=x, key=inputs, value=inputs)  # (batch_size, seq_len, hidden_dim)


        # Add residual connection and feed the output through the FFN
        x = self.ffn(x+output)

        return x

class MLPResNetBlock_V1_4(nn.Module):
    """One MLP ResNet block with a residual connection and Multi-Head Attention conditions."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim

        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        # Multi-head attention layer
        self.multihead_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.CrossAttention = CrossAttention(dim)

    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            h, t, p: (batch_size, 1, hidden_dim) or None
        Returns:
            Output tensor after applying multi-head attention and feedforward network.
        """
        t_inputs = torch.cat([x, h_t], dim=1)  # (batch_size, seq_len + cond_len, hidden_dim)
        t_output, _ = self.multihead_attention(query=x, key=t_inputs, value=t_inputs)  # (batch_size, seq_len, hidden_dim)

        a_output = self.CrossAttention(x, h=h_a, p=p)

        # Add residual connection and feed the output through the FFN
        x = self.ffn(x + t_output + a_output)

        return x

class NoisePredictionModel_V1(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet_V1(
            num_blocks=24,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs, hidden_states = None, time_step=None, proprio_states = None
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        # import pdb; pdb.set_trace()
        output = self.mlp_resnet(obs, h= hidden_states, t = time_step, p = proprio_states)
        return output

# def timestep_embedding(t, dim: int, max_period: int = 10000):
#     # t: (B,) int64/float → (B, dim)
#     import math
#     half = dim // 2
#     freqs = torch.exp(
#         -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
#     )
#     args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
#     emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2 == 1:
#         emb = nn.functional.pad(emb, (0,1))
#     return emb  # (B, dim)

class DiffusionActionHead_V1(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps=100,
    ):
        super().__init__()
        self.action_dim = action_dim
        # self.noise_predictor = NoisePredictionModel_V1(
        #     transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        # )
        self.noise_predictor = NoisePredictionDiT_V1(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=512, action_dim=action_dim
        )
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps, beta_schedule="squaredcos_cap_v2")
        self.num_diffusion_steps = num_diffusion_steps
        # self.time_encoder = SinusoidalPositionalEncoding(dim=512)
        # self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        self.time_encoder = nn.Identity()
        # self.t_embed_dim = 256
        # self.time_encoder = lambda timesteps: timestep_embedding(timesteps, self.t_embed_dim)


    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
            # timesteps=timesteps,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states, noisy_actions=None, timestep_embeddings=None, noisy_action_projector=None, proprio= None, proprio_projector=None):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)

        # import pdb; pdb.set_trace()
        
        if noisy_actions is not None:
            B = noisy_actions.shape[0]
            noisy_actions = noisy_actions.reshape(B, -1).unsqueeze(-1)

            # Project noisy action tokens into language model embedding space
            noise_actions_hidden_states = noisy_action_projector(noisy_actions)  # (B, chunk_len * action_dim, llm_dim)
            # import pdb; pdb.set_trace()
            proprio = proprio.reshape(B, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        batch_size = noise_actions_hidden_states.shape[0]
        device = noise_actions_hidden_states.device
        # import pdb; pdb.set_trace()
        rearranged_actions_hidden_states = noise_actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        # import pdb; pdb.set_trace()
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states, hidden_states = actions_hidden_states, time_step = timestep_embeddings, proprio_states = proprio_features)
        return noise_pred

def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


class FlowMatchingActionHead(nn.Module):
    """
    MLP-based action head that generates continuous actions via flow matching.
    
    Flow matching is an alternative to diffusion models that directly learns the continuous-time flow
    between a simple distribution (e.g., standard normal) and the target distribution.
    
    Inspired by: https://arxiv.org/abs/2210.02747 and PI0: A Vision-Language-Action Flow Model
    """
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_flow_steps=10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_flow_steps = num_flow_steps
        
        # Use the same MLPResNet as the diffusion models but repurposed as a flow predictor
        # self.flow_predictor = NoisePredictionModel(
        #     transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        # )
        self.flow_predictor = FlowPredictionDiT_V1(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=512, action_dim=action_dim
        )

        # Time encoder for positional encoding of timesteps
        # self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        self.time_encoder = nn.Identity()

    def sample_noise(self, shape, device):
        """Sample noise from a standard normal distribution."""
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )
        return noise
    
    # def sample_time(self, bsize, device):
    #     """Sample time values uniformly in [0, 1]."""
    #     # Simple uniform sampling in [0, 1]
    #     time = torch.rand(size=(bsize,), device=device)
    #     return time.to(dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.bfloat16, device=device)
        
    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the flow prediction network. Returns noise, noisy actions, and the
        corresponding flow timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        
        # Sample random noise with shape equal to actions
        noise = self.sample_noise((batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device)
        
        # Sample random flow timesteps (one for each action in batch)
        timesteps = self.sample_time(batch_size, device)
        
        # In flow matching, we interpolate between noise and ground truth based on timestep
        # x_t = (1-t) * noise + t * ground_truth
        time_expanded = timesteps.view(-1, 1, 1)
        noisy_actions = (1 - time_expanded) * noise + time_expanded * ground_truth_actions
        u_t = noise - ground_truth_actions

        timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        timestep_embeddings = timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)
        
        return_dict = dict(
            noise=noise,
            flow=u_t,
            noisy_actions=noisy_actions,
            timestep_embeddings=timestep_embeddings,
        )
        
        return return_dict
    
    def predict_flow(self, actions_hidden_states, noisy_actions=None, timestep_embeddings=None, 
                    noisy_action_projector=None, proprio=None, proprio_projector=None):
        """
        Given a batch of last hidden Transformer layer embeddings, predicts the flow field
        that transforms the noisy actions to the target actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        if noisy_actions is not None:
            noisy_actions_flat = noisy_actions.reshape(batch_size, -1).unsqueeze(-1).to(torch.bfloat16)  # (bsz, chunk_len * action_dim, 1)
            noise_actions_hidden_states = noisy_action_projector(noisy_actions_flat)  # (B, chunk_len * action_dim, llm_dim)
        else:
            noise_actions_hidden_states = torch.zeros_like(actions_hidden_states)
        
        if proprio is not None and proprio_projector is not None:
            proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
            proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)
        else:
            proprio_features = None
        
        rearranged_actions_hidden_states = noise_actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        
        flow_pred = self.flow_predictor(
            obs=rearranged_actions_hidden_states,
            hidden_states=actions_hidden_states,
            time_step=timestep_embeddings,
            proprio_states=proprio_features
        )
        
        return flow_pred
    
    def sample_actions(self, actions_hidden_states, num_steps=10, noisy_action_projector=None, 
                      proprio=None, proprio_projector=None):
        """
        Samples actions by integrating the flow field from noise to the target distribution.
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # Start from standard normal noise
        x = self.sample_noise((batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device)
        
        # Discretize the flow into num_steps
        dt = -1.0 / num_steps  # Negative step size for reverse flow
        
        # Start from t=1 and move backward to t=0
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        # Euler integration of the flow field
        while time >= -dt / 2:
            # Current timestep in [0, 1]
            current_t = time.expand(batch_size)
            
            # Get timestep embeddings
            timestep_embeddings = self.time_encoder(current_t).to(device)
            timestep_embeddings = timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)
            
            # Predict flow at current position
            flow = self.predict_flow(
                actions_hidden_states, 
                noisy_actions=x, 
                timestep_embeddings=timestep_embeddings,
                noisy_action_projector=noisy_action_projector,
                proprio=proprio,
                proprio_projector=proprio_projector
            )
            
            # Update position using Euler step
            x = x + dt * flow
            time += dt
        
        return x