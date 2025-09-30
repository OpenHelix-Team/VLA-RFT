
"""
Single Process World Model
"""

import itertools
from typing import Tuple


import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.workers.world_model import BaseWorldModel
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelWorldModel']


class DataParallelWorldModel(BaseWorldModel):

    def __init__(
        self,
        config,
        world_module: nn.Module,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.world_module = world_module
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.use_old_new_ref = False

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)  #  use torch compile by default
            else verl_F.entropy_from_logits)

    def _forward_micro_batch(self, micro_batch, temperature):
        """
        Returns: 
            When use_old_new_ref=False: log_probs (bs, response_len)
            When use_old_new_ref=True: (gt_logprobs, old_logprobs, new_logprobs, ref_logprobs)
        """
        response_length = micro_batch['gt_responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch.keys():
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            gt_input_ids = micro_batch['gt_input_ids']
            gt_responses = micro_batch['gt_responses']
            batch_size, seqlen = gt_input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if self.use_old_new_ref:
                old_input_ids = micro_batch['old_input_ids']
                new_input_ids = micro_batch['new_input_ids']
                ref_input_ids = micro_batch['ref_input_ids']
                old_responses = micro_batch['old_responses']
                new_responses = micro_batch['new_responses']
                ref_responses = micro_batch['ref_responses']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            # breakpoint()  # for debugging
            if self.use_remove_padding:
                raise NotImplementedError('remove padding not implemented yet')
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.world_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits 
                logits_rmpad = logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div_(temperature)[:, -response_length - 1:-1]  # (bsz, response_length)
            else:  # not using rmpad and no ulysses sp
                if self.use_old_new_ref:
                    gt_output = self.world_module(input_ids=gt_input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            **multi_modal_inputs,
                                            use_cache=False)  # prevent model thinks we are generating
                    gt_logits = gt_output.logits
                    gt_logits.div_(temperature)
                    gt_logits = gt_logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                    gt_logprobs = logprobs_from_logits(gt_logits, gt_responses)

                    old_outputs = self.world_module(input_ids=old_input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            **multi_modal_inputs,
                                            use_cache=False)  # prevent model thinks we are generating
                    old_logits = old_outputs.logits
                    old_logits.div_(temperature)
                    old_logits = old_logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                    # breakpoint()
                    old_logprobs = logprobs_from_logits(old_logits, old_responses)

                    new_outputs = self.world_module(input_ids=new_input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            **multi_modal_inputs,
                                            use_cache=False)  # prevent model thinks we are generating
                    new_logits = new_outputs.logits
                    new_logits.div_(temperature)
                    new_logits = new_logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                    new_logprobs = logprobs_from_logits(new_logits, new_responses)

                    ref_outputs = self.world_module(input_ids=ref_input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            **multi_modal_inputs,
                                            use_cache=False)  # prevent model thinks we are generating
                    ref_logits = ref_outputs.logits
                    ref_logits.div_(temperature)
                    ref_logits = ref_logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                    ref_logprobs = logprobs_from_logits(ref_logits, ref_responses)

                    return gt_logprobs, old_logprobs, new_logprobs, ref_logprobs
                else:
                    output = self.world_module(input_ids=gt_input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            **multi_modal_inputs,
                                            use_cache=False)  # prevent model thinks we are generating
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, gt_responses)
                    # entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                    return log_probs
    def _forward_micro_batch_compute_gt_logits(self, micro_batch, temperature) -> torch.Tensor:
        """
        Returns: 
            gt_logits: # (bs, response_len, vocab_size)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch.keys():
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            gt_seq = micro_batch['gt_seq']
            # batch_size, seqlen = gt_seq.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)
            # breakpoint()  # for debugging
            if self.use_remove_padding:
                raise NotImplementedError("GT logits computation does not support remove padding yet.")
            else:  # not using rmpad and no ulysses sp
                output = self.world_module(input_ids=gt_seq,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                gt_logits = output.logits
                gt_logits.div_(temperature)
                gt_logits = gt_logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)

            return gt_logits

    def compute_log_prob(self, data: DataProto) -> DataProto:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            DataProto: containing log probability tensors
                When use_old_new_ref=False: {'gt_log_probs': torch.Tensor}
                When use_old_new_ref=True: {'gt_log_probs': torch.Tensor, 'old_log_probs': torch.Tensor, 'new_log_probs': torch.Tensor, 'ref_log_probs': torch.Tensor}
        """
        # set to eval
        self.world_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['gt_responses', 'gt_input_ids', 'attention_mask', 'position_ids']
        self.use_old_new_ref = True if 'old_input_ids' in data.batch.keys() and 'new_input_ids' in data.batch.keys() and \
                        'ref_input_ids' in data.batch.keys() else False
        if self.use_old_new_ref:
            select_keys.extend(['old_input_ids', 'new_input_ids', 'ref_input_ids', 'old_responses', 'new_responses', 'ref_responses'])
        
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)
        
        if self.use_old_new_ref:
            # if use old/new/ref, we need to compute gt_log_probs, old_log_probs, new_log_probs, ref_log_probs
            gt_log_probs_lst = []
            old_log_probs_lst = []
            new_log_probs_lst = []
            ref_log_probs_lst = []
        else:
            gt_log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            # breakpoint()
            if self.use_old_new_ref:
                with torch.no_grad():
                    gt_log_probs, old_log_probs, new_log_probs, ref_log_probs = self._forward_micro_batch(micro_batch, temperature)
                gt_log_probs_lst.append(gt_log_probs)
                old_log_probs_lst.append(old_log_probs)
                new_log_probs_lst.append(new_log_probs)
                ref_log_probs_lst.append(ref_log_probs)
            else:
                with torch.no_grad():
                    log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
                gt_log_probs_lst.append(log_probs)
        if self.use_old_new_ref:
            gt_log_probs = torch.concat(gt_log_probs_lst, dim=0)
            old_log_probs = torch.concat(old_log_probs_lst, dim=0)
            new_log_probs = torch.concat(new_log_probs_lst, dim=0)
            ref_log_probs = torch.concat(ref_log_probs_lst, dim=0)
            output = DataProto.from_single_dict({
                'gt_log_probs': gt_log_probs,
                'old_log_probs': old_log_probs,
                'new_log_probs': new_log_probs,
                'ref_log_probs': ref_log_probs
            })
        else:
            log_probs = torch.concat(gt_log_probs_lst, dim=0)
            output = DataProto.from_single_dict({
                'gt_log_probs': log_probs
            })

        if use_dynamic_bsz:
            raise NotImplementedError("Dynamic batch size is not supported in DataParallelWorldModel.")
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return output

