# -*- coding: utf-8 -*-
#
# Copyright 2023, 2024 Ramil Nugmanov <nougmanoff@protonmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from math import sqrt
from torch import softmax, cat, Tensor
from torch.nn import Module
from torch.nn.functional import dropout
from typing import Optional, Tuple
from warnings import warn
from ...lora import Linear


def _update_unpacked(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'in_proj_weight' in state_dict:
        warn('fixed chytorch<1.44 checkpoint', DeprecationWarning)
        state_dict[prefix + 'qkv_proj.weight'] = state_dict.pop(prefix + 'in_proj_weight')
        state_dict[prefix + 'qkv_proj.bias'] = state_dict.pop(prefix + 'in_proj_bias')
        state_dict[prefix + 'o_proj.weight'] = state_dict.pop(prefix + 'out_proj.weight')
        state_dict[prefix + 'o_proj.bias'] = state_dict.pop(prefix + 'out_proj.bias')

    if prefix + 'qkv_proj.weight' in state_dict:  # transform packed projection
        q_w, k_w, v_w = state_dict.pop(prefix + 'qkv_proj.weight').chunk(3, dim=0)
        state_dict[prefix + 'q_proj.weight'] = q_w
        state_dict[prefix + 'k_proj.weight'] = k_w
        state_dict[prefix + 'v_proj.weight'] = v_w

        if prefix + 'qkv_proj.bias' in state_dict:
            q_b, k_b, v_b = state_dict.pop(prefix + 'qkv_proj.bias').chunk(3, dim=0)
            state_dict[prefix + 'q_proj.bias'] = q_b
            state_dict[prefix + 'k_proj.bias'] = k_b
            state_dict[prefix + 'v_proj.bias'] = v_b


def _update_packed(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'in_proj_weight' in state_dict:
        warn('fixed chytorch<1.44 checkpoint', DeprecationWarning)
        state_dict[prefix + 'o_proj.weight'] = state_dict.pop(prefix + 'out_proj.weight')
        state_dict[prefix + 'o_proj.bias'] = state_dict.pop(prefix + 'out_proj.bias')

        state_dict[prefix + 'qkv_proj.weight'] = state_dict.pop(prefix + 'in_proj_weight')
        state_dict[prefix + 'qkv_proj.bias'] = state_dict.pop(prefix + 'in_proj_bias')
    elif prefix + 'q_proj.weight' in state_dict:  # transform unpacked projection
        q_w = state_dict.pop(prefix + 'q_proj.weight')
        k_w = state_dict.pop(prefix + 'k_proj.weight')
        v_w = state_dict.pop(prefix + 'v_proj.weight')
        state_dict[prefix + 'qkv_proj.weight'] = cat([q_w, k_w, v_w])

        if prefix + 'q_proj.bias' in state_dict:
            q_b = state_dict.pop(prefix + 'q_proj.bias')
            k_b = state_dict.pop(prefix + 'k_proj.bias')
            v_b = state_dict.pop(prefix + 'v_proj.bias')
            state_dict[prefix + 'qkv_proj.bias'] = cat([q_b, k_b, v_b])


class GraphormerAttention(Module):
    """
    LoRA wrapped Multi-Head Attention
    """
    def __init__(self, embed_dim, num_heads, dropout: float = .1, bias: bool = True, separate_proj: bool = False):
        """
        :param embed_dim: the size of each embedding vector
        :param num_heads: number of heads
        :param dropout: attention dropout
        :param separate_proj: use separated projections calculations or optimized
        """
        assert not embed_dim % num_heads, 'embed_dim must be divisible by num_heads'
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.separate_proj = separate_proj
        self._scale = 1 / sqrt(embed_dim / num_heads)

        if separate_proj:
            self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
            self._register_load_state_dict_pre_hook(_update_unpacked)
        else:  # packed projection
            self.qkv_proj = Linear(embed_dim, 3 * embed_dim, bias=bias)
            self._register_load_state_dict_pre_hook(_update_packed)
        self.o_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: Tensor, attn_mask: Tensor, *,
                cache: Optional[Tuple[Tensor, Tensor]] = None,
                need_weights: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        if self.separate_proj:
            q = self.q_proj(x)  # BxTxH*E
            k = self.k_proj(x)  # BxSxH*E (KV seq len can differ from tgt_len with enabled cache trick)
            v = self.v_proj(x)  # BxSxH*E
        else:  # optimized
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        if cache is not None:
            # inference caching. batch should be left padded. shape should be BxSxH*E
            bsz, tgt_len, _ = x.shape
            ck, cv = cache
            ck[:bsz, -tgt_len:] = k
            cv[:bsz, -tgt_len:] = v
            k, v = ck[:bsz], cv[:bsz]

        # BxTxH*E > BxTxHxE > BxHxTxE
        q = q.unflatten(2, (self.num_heads, -1)).transpose(1, 2)
        # BxSxH*E > BxSxHxE > BxHxExS
        k = k.unflatten(2, (self.num_heads, -1)).permute(0, 2, 3, 1)
        # BxSxH*E > BxSxHxE > BxHxSxE
        v = v.unflatten(2, (self.num_heads, -1)).transpose(1, 2)

        # BxHxTxE @ BxHxExS > BxHxTxS
        a = (q @ k) * self._scale + attn_mask
        a = softmax(a, dim=-1)
        if self.training and self.dropout:
            a = dropout(a, self.dropout)

        # BxHxTxS @ BxHxSxE > BxHxTxE > BxTxHxE > BxTxH*E
        o = (a @ v).transpose(1, 2).flatten(2)
        o = self.o_proj(o)

        if need_weights:
            a = a.sum(dim=1) / self.num_heads
            return o, a
        else:
            return o, None


__all__ = ['GraphormerAttention']
