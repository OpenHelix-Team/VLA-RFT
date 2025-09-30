import time
import torch
import numpy as np
import imageio
import math
from verl.utils.model import compute_position_id_with_mask


def batch_forward(batch_size, input, forward):
    return torch.cat([forward(input[i: i + batch_size]) for i in range(0, input.shape[0], batch_size)], dim=0)


def batch_forward2(batch_size, input1, input2, forward):
    return torch.cat([forward(input1[i: i + batch_size], input2[i: i + batch_size]) for i in range(0, input1.shape[0], batch_size)], dim=0)


def batch_forward3(batch_size, input, forward):
    output1, output2 = [], []
    for i in range(0, input.shape[0], batch_size):
        out1, out2 = forward(input[i: i + batch_size])
        output1.append(out1)
        output2.append(out2)
    return torch.cat(output1, dim=0), torch.cat(output2, dim=0)


def plot_gif(x, postfix=''):
    # [B, T, C, H, W]
    frames = [(x[0, i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8) for i in range(x.shape[1])]
    imageio.mimsave(f"tmp{postfix}.gif", frames, fps=4, loop=0)


class SimpleVideoProcessor:
    def __init__(self, config, visual_tokenizer):
        self.config = config
        self.visual_tokenizer = visual_tokenizer
        self.action_ranges = torch.load(config.action_ranges_path)  # [num_actions, 2]

    def _discretize_actions(self, actions, num_bins=256):
        if actions.dim() == 3:
            b, t = actions.shape[:2]
            actions = actions.reshape(b * t, -1)
        else:
            b, t = None, None

        max_values, min_values = self.action_ranges[:, 1], self.action_ranges[:, 0]
        actions = torch.clip((actions - min_values) / (max_values - min_values + 1e-8), 0, 1)
        actions = torch.floor(actions * num_bins).to(torch.int32).clip(0, num_bins - 1)

        if b is not None and t is not None:
            actions = actions.reshape(b, t, *actions.shape[1:])
        return actions

    @torch.no_grad()
    def detokenize(self, tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.config.tokenizer_micro_batch_size is not None:
                return batch_forward(
                    self.config.tokenizer_micro_batch_size, tokens,
                    lambda x: self.visual_tokenizer.decode(x),
                )
            else:
                return self.visual_tokenizer.decode(tokens)

    def _create_response(self, ground_truth_tokens):
        """
        ground_truth_tokens: [B, T, N]
        """
        device = ground_truth_tokens.device
        b, t, n = ground_truth_tokens.shape

        input_ids = torch.cat([
            torch.ones((b, t, 1), dtype=ground_truth_tokens.dtype, device=device) * self.config.bos_token_id,
            ground_truth_tokens
        ], dim=2).reshape(b, -1)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.float32, device=device)
        loss_mask = torch.ones(input_ids.shape, dtype=torch.float32, device=device)

        singles = lambda value, type: torch.ones((b, 1), device=device, dtype=type) * value

        # add eos id
        input_ids = torch.cat([input_ids, singles(self.config.eos_token_id, input_ids.dtype)], dim=1)
        attention_mask = torch.cat([attention_mask, singles(1.0, attention_mask.dtype)], dim=1)
        loss_mask = torch.cat([loss_mask, singles(1.0, loss_mask.dtype)], dim=1)

        labels = input_ids

        return input_ids, attention_mask, loss_mask, labels

    @torch.no_grad()
    def __call__(self, pixels, actions, return_interpolated=False):
        """
        pixels: (B, T, C, H, W)
        actions: (B, T, num_actions)
        """
        self.action_ranges = self.action_ranges.to(pixels.device)
        context_length = self.config.context_length

        b, t = pixels.shape[:2]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.config.tokenizer_micro_batch_size is not None:
                pixel_tokens = batch_forward(
                    self.config.tokenizer_micro_batch_size, pixels,
                    lambda x: self.visual_tokenizer.encode(x).reshape(*x.shape[:2], -1),
                )
            else:
                pixel_tokens = self.visual_tokenizer.encode(pixels).reshape(b, t, -1)  # [B, T, h*w]

        hist_pixel_tokens = pixel_tokens[:, :context_length]  # [B, T-1, h*w]
        action_tokens = self._discretize_actions(
            actions[:, :context_length], self.config.action_bins)  # [B, T-1, num_actions]
        action_tokens += self.config.visual_token_num  # offset action tokens
        hist_tokens = torch.cat([
            hist_pixel_tokens, action_tokens], dim=-1).reshape(b, -1)  # [B, (T-1)*(h*w+num_actions)]

        input_ids, attention_mask, loss_mask, labels = self._create_response(
            pixel_tokens[:, context_length:],
        )
        labels = torch.cat([torch.ones_like(hist_tokens) * -100, input_ids], dim=-1)
        input_ids = torch.cat([hist_tokens, input_ids], dim=-1)  # [B, N]
        attention_mask = torch.cat([torch.ones_like(hist_tokens).to(attention_mask.dtype), attention_mask], dim=-1)
        loss_mask = torch.cat([loss_mask[:, 1:], torch.zeros((b, 1), device=loss_mask.device)], dim=-1)

        position_ids = compute_position_id_with_mask(attention_mask)
        loss_mask = torch.cat([torch.zeros_like(hist_tokens).to(loss_mask.dtype), loss_mask], dim=-1)

        output_dict = {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            # 'loss_mask': loss_mask,
            'labels': labels.long(),
        }

        if return_interpolated:
            return output_dict, pixels
        else:
            return output_dict


class ContextMultiStepPredictionProcessor:
    def __init__(self, config, visual_tokenizer):
        self.config = config
        self.visual_tokenizer = visual_tokenizer
        self.action_ranges = torch.load(config.action_ranges_path)  # [num_actions, 2]

    def _discretize_actions(self, actions, num_bins=256):
        if actions.dim() == 3:
            b, t = actions.shape[:2]
            actions = actions.reshape(b * t, -1)
        else:
            b, t = None, None

        max_values, min_values = self.action_ranges[:, 1], self.action_ranges[:, 0]
        actions = torch.clip((actions - min_values) / (max_values - min_values + 1e-8), 0, 1)
        actions = torch.floor(actions * num_bins).to(torch.int32).clip(0, num_bins - 1)

        if b is not None and t is not None:
            actions = actions.reshape(b, t, *actions.shape[1:])
        return actions

    @torch.no_grad()
    def detokenize(self, ctx_tokens, tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.config.tokenizer_micro_batch_size is not None:
                return batch_forward2(
                    self.config.tokenizer_micro_batch_size, ctx_tokens, tokens,
                    lambda x, y: self.visual_tokenizer.detokenize(x, y),
                )
            else:
                return self.visual_tokenizer.detokenize(ctx_tokens, tokens)

    @torch.no_grad()
    def __call__(self, pixels, actions, return_interpolated=False, return_ctx_tokens=False): 
        """
        pixels: (B, T+1, C, H, W)
        actions: (B, T+1, num_actions)
        """
        self.action_ranges = self.action_ranges.to(pixels.device)

        b, _ = pixels.shape[:2]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # ctx_tokens, dyn_tokens = self.visual_tokenizer.tokenize(pixels)  # [B, 1, H*W], [B, T, h*w]
            if self.config.tokenizer_micro_batch_size is not None:
                ctx_tokens, dyn_tokens = batch_forward3(
                    self.config.tokenizer_micro_batch_size, pixels,
                    lambda x: self.visual_tokenizer.tokenize(x),
                )
            else:
                ctx_tokens, dyn_tokens = self.visual_tokenizer.tokenize(pixels)
            # recon = self.visual_tokenizer.detokenize(ctx_tokens, dyn_tokens).clip(0, 1)
        ctx_tokens += self.config.visual_token_num # offset context tokens
        hist_dyn_tokens = dyn_tokens  # [B, T, h*w]
        # breakpoint()
        action_tokens = self._discretize_actions(actions[:, 1:], self.config.action_bins)  # [B, T, num_actions]
        action_tokens += self.config.visual_token_num * 2  # offset action tokens
        
        
        hist_tokens = torch.cat([hist_dyn_tokens, action_tokens], dim=-1).reshape(b, -1)  # [B, T*(h*w+num_actions)]
        input_ids = torch.cat([ctx_tokens.reshape(b, -1), hist_tokens], dim=-1)  # [B, H*W + T*(h*w+num_actions)]
        
        labels = hist_tokens
        labels[:, :hist_dyn_tokens.shape[-1]] = -100
        labels = torch.cat([torch.ones_like(ctx_tokens.reshape(b, -1)) * -100, labels], dim=-1)
        
        attention_mask = torch.ones_like(input_ids).to(torch.float32)
        position_ids = compute_position_id_with_mask(attention_mask)
        
        output_dict = {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels.long(),
            "action_ids": action_tokens.long(),
        }

        if return_interpolated:
            if return_ctx_tokens:
                return output_dict, pixels, ctx_tokens
            else:
                return output_dict, pixels
        else:
            if return_ctx_tokens:
                return output_dict, ctx_tokens
            else:
                return output_dict


PROCESSOR_TYPE = {
    'simple': SimpleVideoProcessor,
    'ctx_msp': ContextMultiStepPredictionProcessor,
}
