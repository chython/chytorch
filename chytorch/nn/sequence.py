# -*- coding: utf-8 -*-
#
#  Copyright 2022 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from math import log, inf
from torch import Tensor, cat, zeros, arange, exp, sin, cos, ones, triu, zeros_like, float as t_float
from torch.nn import Dropout, GELU, Linear, LayerNorm, Module, MultiheadAttention, Embedding, ModuleList
from torchtyping import TensorType


class SequenceDecoderLayer(Module):
    """
    Inspired by torch.nn.modules.transformer.TransformerDecoderLayer layer
     adopted for an aggregated latent space to sequence decoding.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation=GELU, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation()

    def forward(self, x: Tensor, e: Tensor, attn_mask: Tensor) -> Tensor:
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]))
        x = self.norm2(x + e)  # MHA replaced by shared encoder embedding
        return self.norm3(x + self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x))))))


class SequenceDecoder(Module):
    """
    Transformer decoder adopted to generate sequences from molecular embeddings.
    """
    def __init__(self, dict_size: int = 100, seq_max_len: int = 100, shared_weights: bool = True,
                 d_model: int = 1024, nhead: int = 16, num_layers: int = 8, dim_feedforward: int = 3072,
                 dropout: float = 0.1, activation=GELU, layer_norm_eps: float = 1e-5):
        """
        Sequence TransformerDecoder layer.

        :param dict_size: number of possible tokens.
        :param seq_max_len: maximal sequence length to decode.
        :param shared_weights: ALBERT-like encoder weights sharing.
        """
        super().__init__()
        self.embedding = Embedding(dict_size, d_model, padding_idx=0)
        self.linear = Linear(d_model, dict_size)
        self.dropout = Dropout(dropout)

        if shared_weights:
            self.layer = SequenceDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
            self.layers = [self.layer] * num_layers
        else:
            self.layers = ModuleList(SequenceDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                          layer_norm_eps)
                                     for _ in range(num_layers))

        self.register_buffer('pos_encoding', _pos_encoding(seq_max_len, d_model), persistent=False)
        self.nhead = nhead

    def forward(self, emb: TensorType['batch', 'embedding'], seq: TensorType['batch', 'tokens', int]) -> \
            TensorType['batch', 'tokens', 'logits']:
        n = seq.size(1)
        p_mask = zeros_like(seq, dtype=t_float).masked_fill_(seq == 0, -inf).unsqueeze_(1). \
            masked_fill(triu(ones(n, n, dtype=bool, device=seq.device), diagonal=1), -inf).unsqueeze_(1). \
            expand(-1, self.nhead, -1, -1).flatten(end_dim=1)

        x = self.dropout(self.embedding(seq) + self.pos_encoding[:n])  # positionally coded sequence
        e = emb.unsqueeze(1).expand(x.size())
        for lr in self.layers:
            x = lr(x, e, p_mask)
        return self.linear(x)


def _pos_encoding(max_len, d_model):
    # stolen from: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    pos_encoding = zeros(max_len, d_model)
    positions_list = arange(0, max_len).float().view(-1, 1)
    division_term = exp(arange(0, d_model, 2).float() * -log(10000.) / d_model)  # 1000^(2i/dim_model)
    # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
    pos_encoding[:, 0::2] = sin(positions_list * division_term)
    # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
    pos_encoding[:, 1::2] = cos(positions_list * division_term)
    return pos_encoding


__all__ = ['SequenceDecoder']
