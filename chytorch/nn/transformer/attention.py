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
from torch import softmax, cat, Tensor
from torch.nn import Module
from torch.nn.functional import dropout
from typing import Optional, Tuple
from warnings import warn
from ..lora import Linear


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
    def __init__(self, embed_dim, num_heads, dropout=.1, separate_proj: bool = False,
                 lora_r: int = 0, lora_alpha: float = 1., lora_dropout: float = 0.):
        """
        :param embed_dim: the size of each embedding vector
        :param num_heads: number of heads
        :param dropout: attention dropout
        :param separate_proj: use separated projections calculations or optimized
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
        self.separate_proj = separate_proj or bool(lora_r)
        self._scale = 1 / sqrt(self.head_dim)

        if separate_proj or lora_r:
            self.q_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.k_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.v_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self._register_load_state_dict_pre_hook(_update_lora)
        else:  # packed projection
            self.qkv_proj = Linear(embed_dim, 3 * embed_dim)
            self._register_load_state_dict_pre_hook(_update_packed)
        self.o_proj = Linear(embed_dim, embed_dim, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor], pad_mask: Optional[Tensor] = None, *,
                cache: Optional[Tuple[Tensor, Tensor]] = None,
                need_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, tgt_len, _ = x.shape

        # do projection
        if self.separate_proj:
            q = self.q_proj(x)  # BxTxH*E
            k = self.k_proj(x)  # BxSxH*E (KV seq len can differ from tgt_len with enabled cache trick)
            v = self.v_proj(x)  # BxSxH*E
        else:  # optimized
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        if cache is not None:
            # inference caching. batch should be left padded. shape should be BxSxH*E
            ck, cv = cache
            ck[:bsz, -tgt_len:] = k
            cv[:bsz, -tgt_len:] = v
            k, v = ck[:bsz], cv[:bsz]

        # BxTxH*E > BxTxHxE > BxHxTxE
        q = q.reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # BxSxH*E > BxSxHxE > BxHxExS
        k = k.reshape(bsz, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        # BxSxH*E > BxSxHxE > BxHxSxE
        v = v.reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # BxHxTxE @ BxHxExS > BxHxTxS
        a = (q @ k) * self._scale
        if attn_mask is not None:
            a = a + attn_mask
        a = softmax(a, dim=-1)
        if self.training and self.dropout:
            a = dropout(a, self.dropout)

        # BxHxTxS @ BxHxSxE > BxHxTxE > BxTxHxE > BxTxH*E
        o = (a @ v).transpose(1, 2).flatten(start_dim=2)
        o = self.o_proj(o)

        if need_weights:
            a = a.sum(dim=1) / self.num_heads
            return o, a
        else:
            return o, None

    def merge_lora(self):
        """
        Transform LoRA MHA to normal
        """
        if not self.lora_r:
            return
        self.q_proj.merge_lora()
        self.k_proj.merge_lora()
        self.v_proj.merge_lora()
        self.o_proj.merge_lora()


__all__ = ['MultiheadAttention']
