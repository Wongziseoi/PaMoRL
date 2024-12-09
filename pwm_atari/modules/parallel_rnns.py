import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import scan
import modules.networks as net


class RNNCell(nn.Module):
    def __init__(self, inp_size, hidden, act, divisor=4, pdrop=0.1):
        super().__init__()
        self.rnn = ParaRNNLayer(inp_size, hidden, divisor)
        self.ffn = net.GatingLayer(inp_size, act())
        self.norm = nn.LayerNorm(inp_size)
        self.drop = nn.Dropout(pdrop)

    @torch.no_grad()
    def initial(self, batch_size, layer_id):
        inp, state = self.rnn.initial_state(batch_size)
        return {f"inp_{layer_id}": inp, f"rnn_{layer_id}": state}
    
    def initial_stats(self, init, layer_id):
        init_state = init[f"rnn_{layer_id}"]
        init_deter = self.rnn.proj(init_state)
        return init_deter

    def forward(self, inp, is_first, state, parallel, layer_id):
        drop_mask = self.drop(torch.ones_like(inp[..., :1]))
        states = (state[f"inp_{layer_id}"], state[f"rnn_{layer_id}"])

        output, last_input, internal_state = self.rnn(
            inp, drop_mask, is_first, states, parallel)
        output = self.norm(inp + self.drop(output))

        output = self.ffn(output)
        
        stats = {
            f"inp_{layer_id}": last_input, 
            f"rnn_{layer_id}": internal_state
        }
        return output, stats


class ParaRNNLayer(nn.Module):
    def __init__(self, inp_size, hidden, divisor):
        super().__init__()
        self.inp_size = inp_size
        assert hidden % divisor == 0
        self.query_dim = hidden // divisor
        self.embed_dim = hidden // divisor

        concat_size = 2 * (self.query_dim + self.embed_dim)
        self.layer = nn.Linear(inp_size, concat_size, bias=False)
        self.gate = net.MixingLayer(inp_size, self.query_dim, bias=False)
        self.proj = nn.Linear(self.embed_dim, inp_size, bias=False)
        self.norm = net.RMSNorm(self.embed_dim)

    def initial_state(self, batch_size):
        hidden_size = self.query_dim * self.embed_dim
        return (
            torch.zeros(batch_size, self.inp_size), # last inputs
            torch.zeros(batch_size, hidden_size),
        )
    
    def components(self, x, last, mask):
        forget = self.gate(x, last)
        forget = torch.sigmoid(forget)
        forget = (forget - 1) * mask + 1

        parts = self.layer(x) * mask
        query_key, reset_value = parts.split(
            [self.query_dim * 2, self.embed_dim * 2], dim=-1)
        query_key = F.elu(query_key) + 1
        query, key = query_key.chunk(2, dim=-1)

        reset, value = reset_value.chunk(2, dim=-1)
        value = torch.sigmoid(reset) * value
        cand = key[..., None] * value[..., None, :]
        forget = forget[..., None].expand(*cand.shape)
        return forget, cand, query

    def forward(self, inputs, drops, is_first=None, state=None, parallel=False):
        return self._parallel_forward(inputs, drops, is_first, state) \
            if parallel else self._recurrent_forward(inputs, drops, state)

    def _parallel_forward(self, inputs, drops, is_first, inits):
        init_inp, init = inits
        init_size = (self.query_dim, self.embed_dim)
        init = init[None, ...].unflatten(-1, init_size)
        ones = torch.ones_like(is_first[:1])
        init_masks = torch.cat((ones, is_first[1:]), dim=0)
        
        lasts = F.pad(inputs, (0, 0, 0, 0, 1, -1), "constant", 0)
        lasts += init_masks * (init_inp[None, ...] - lasts)
        
        components = self.components(inputs, lasts, drops)
        forgets, cands, querys = components
        cands = cands + init_masks[..., None] * forgets * init

        masks = is_first.view(*cands.shape[:2], 1, 1).expand(*cands.shape)
        states = scan.odd_even_parallel_scan(
            [forgets, cands, masks], binary_operator)[1]

        outputs = querys[..., None, :] @ states
        outputs = self.proj(self.norm(outputs.squeeze(-2)))
        return outputs, inputs, states.flatten(-2, -1)

    def _recurrent_forward(self, input, drop, states):
        last, state = states
        state_size = (self.query_dim, self.embed_dim)
        state = state.unflatten(-1, state_size)

        component = self.components(input, last, drop)
        forget, cand, query = component
        state = compute_seq_ssm(forget, state, cand)
        
        output = query[..., None, :] @ state
        output = self.proj(self.norm(output.squeeze(-2)))
        return output, input, state.flatten(-2, -1)


def binary_operator(element_i, element_j):
    update_i, input_i, mask_i = element_i
    update_j, input_j, mask_j = element_j

    condition = mask_j > 0
    new_update_j = torch.where(
        condition, update_j, update_j * update_i)
    new_input_j = torch.where(
        condition, input_j, update_j * input_i + input_j)
    new_mask_j = torch.where(condition, mask_j, mask_i)
    return new_update_j, new_input_j, new_mask_j


def compute_seq_ssm(forget, state, cand):
    return forget * state + cand