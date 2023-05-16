# -*- coding: utf-8 -*-
#
#  Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of chytorch.
#
#  chytorch is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from math import sqrt
from torch import baddbmm, bmm, softmax, cat, Tensor
from torch.nn import Module
from torch.nn.functional import dropout, linear
from typing import Optional, Tuple
from warnings import warn
from .linear import Linear


def _update_lora(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'in_proj_weight' in state_dict:
        warn('fixed chytorch<1.44 checkpoint', DeprecationWarning)
        state_dict[prefix + 'o_proj.weight'] = state_dict.pop(prefix + 'out_proj.weight')
        state_dict[prefix + 'o_proj.bias'] = state_dict.pop(prefix + 'out_proj.bias')

        q_w, k_w, v_w = state_dict.pop(prefix + 'in_proj_weight').chunk(3, dim=0)
        q_b, k_b, v_b = state_dict.pop(prefix + 'in_proj_bias').chunk(3, dim=0)
        state_dict[prefix + 'q_proj.weight'] = q_w
        state_dict[prefix + 'k_proj.weight'] = k_w
        state_dict[prefix + 'v_proj.weight'] = v_w
        state_dict[prefix + 'q_proj.bias'] = q_b
        state_dict[prefix + 'k_proj.bias'] = k_b
        state_dict[prefix + 'v_proj.bias'] = v_b
    elif prefix + 'qkv_proj.weight' in state_dict:  # transform packed projection
        q_w, k_w, v_w = state_dict.pop(prefix + 'qkv_proj.weight').chunk(3, dim=0)
        q_b, k_b, v_b = state_dict.pop(prefix + 'qkv_proj.bias').chunk(3, dim=0)
        state_dict[prefix + 'q_proj.weight'] = q_w
        state_dict[prefix + 'k_proj.weight'] = k_w
        state_dict[prefix + 'v_proj.weight'] = v_w
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
        q_b = state_dict.pop(prefix + 'q_proj.bias')
        k_b = state_dict.pop(prefix + 'k_proj.bias')
        v_b = state_dict.pop(prefix + 'v_proj.bias')
        state_dict[prefix + 'qkv_proj.weight'] = cat([q_w, k_w, v_w])
        state_dict[prefix + 'qkv_proj.bias'] = cat([q_b, k_b, v_b])


class MultiheadAttention(Module):
    """
    LoRA wrapped Multi-Head Attention
    """
    def __init__(self, embed_dim, num_heads, dropout=0., lora_r: int = 0, lora_alpha: float = 1.,
                 lora_dropout: float = 0.):
        """
        :param embed_dim: the size of each embedding vector
        :param num_heads: number of heads
        :param dropout: attention dropout
        :param lora_r: LoRA factorization dimension
        :param lora_alpha: LoRA scaling factor
        :param lora_dropout: LoRA input dropout
        """
        assert not embed_dim % num_heads, 'embed_dim must be divisible by num_heads'
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.lora_r = lora_r
        self._scale = 1 / sqrt(self.head_dim)

        if lora_r:
            self.q_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.k_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.v_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self._register_load_state_dict_pre_hook(_update_lora)
        else:  # packed projection
            self.qkv_proj = Linear(embed_dim, 3 * embed_dim)
            self._register_load_state_dict_pre_hook(_update_packed)
        self.o_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, query: Tensor, key_value: Tensor, attn_mask: Tensor,
                need_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, tgt_len, _ = query.shape
        _, src_len, _ = key_value.shape
        the_same = query is key_value

        query = query.transpose(1, 0)  # switch batch and sequence dims

        # do projection
        if self.lora_r:
            key_value = key_value.transpose(1, 0)

            q = self.q_proj(query)
            k = self.k_proj(key_value)
            v = self.v_proj(key_value)
        elif the_same:  # self-attention
            q, k, v = self.qkv_proj(query).chunk(3, dim=-1)
        else:  # cross-attention
            key_value = key_value.transpose(1, 0)

            w_q, w_kv = self.qkv_proj.weight.split([self.embed_dim, self.embed_dim * 2])
            b_q, b_kv = self.qkv_proj.bias.split([self.embed_dim, self.embed_dim * 2])
            q = linear(query, w_q, b_q)
            k, v = linear(key_value, w_kv, b_kv).chunk(2, dim=-1)

        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.reshape(src_len, bsz * self.num_heads, self.head_dim).permute(1, 2, 0)
        v = v.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        a = baddbmm(attn_mask, q, k, alpha=self._scale)  # scaled dot-product with bias
        a = softmax(a, dim=-1)
        if self.training and self.dropout:
            a = dropout(a, self.dropout)

        o = bmm(a, v).transpose(0, 1).contiguous().view(tgt_len * bsz, self.embed_dim)
        o = self.o_proj(o).view(tgt_len, bsz, -1).transpose(0, 1)  # switch dimensions back

        if need_weights:
            a = a.view(bsz, self.num_heads, tgt_len, src_len)
            a = a.sum(dim=1) / self.num_heads
            return o, a
        else:
            return o, None

    def merge_lora(self):
        """
        Transform LoRA MHA to normal
        """
        self.q_proj.merge_lora()
        self.k_proj.merge_lora()
        self.v_proj.merge_lora()
        self.o_proj.merge_lora()


__all__ = ['MultiheadAttention']
