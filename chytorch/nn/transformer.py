# -*- coding: utf-8 -*-
#
#  Copyright 2021, 2022 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from torch import Tensor
from torch.nn import Dropout, GELU, Linear, LayerNorm, Module, MultiheadAttention
from typing import Tuple, Optional


class EncoderLayer(Module):
    r"""EncoderLayer based on torch.nn.TransformerEncoderLayer, but batch always first and returns also attention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (required).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer. Default: GELU.
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation=GELU, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation()

    def forward(self, x: Tensor, attn_mask: Tensor, *,
                need_embedding: bool = True, need_weights: bool = False) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        tx = x.transpose(1, 0)  # switch Batch and Sequence. torch-1.8 compatible.
        e, a = self.self_attn(tx, tx, tx, attn_mask=attn_mask, need_weights=need_weights)
        if need_embedding:
            x = self.norm1(tx + self.dropout1(e)).transpose(1, 0)  # switch Sequence and Batch
            return self.norm2(x + self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x)))))), a
        return None, a


class DecoderLayer(Module):
    r"""DecoderLayer based on torch.nn.TransformerDecoderLayer, but batch always first and returns also attention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (required).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer. Default: GELU.
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation=GELU, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.tgt_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)
        self.activation = activation()

    def forward(self, x: Tensor, mem: Tensor, self_attn_mask: Tensor, target_attn_mask: Tensor, *,
                disable_self_attention: bool = False, need_embedding: bool = True, need_weights: bool = False) -> \
            Tuple[Optional[Tensor], Optional[Tensor]]:
        tx = x.transpose(1, 0)  # switch Batch and Sequence. torch-1.8 compatible.
        tm = mem.transpose(1, 0)
        if not disable_self_attention:
            tx = self.norm1(tx + self.dropout1(self.self_attn(tx, tx, tx, attn_mask=self_attn_mask,
                                                              need_weights=False)[0]))
        e, a = self.tgt_attn(tx, tm, tm, attn_mask=target_attn_mask, need_weights=need_weights)
        if need_embedding:
            x = self.norm2(tx + self.dropout2(e)).transpose(1, 0)  # switch Sequence and Batch
            return self.norm3(x + self.dropout4(self.linear2(self.dropout3(self.activation(self.linear1(x)))))), a
        return None, a


__all__ = ['EncoderLayer', 'DecoderLayer']
