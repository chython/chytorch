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
from torch import stack
from torch.nn import GELU, Module, ModuleList, LayerNorm, Dropout
from torchtyping import TensorType
from typing import Tuple, Union
from ..molecule import MoleculeEncoder
from ..transformer import DecoderLayer


class ReactionDecoder(Module):
    def __init__(self, max_neighbors: int = 14, max_distance: int = 10, d_model: int = 1024, n_in_head: int = 16,
                 n_ex_head: int = 16, shared_in_weights: bool = True, shared_ex_weights: bool = True,
                 shared_layers: bool = False, num_in_layers: int = 8, num_ex_layers: int = 8,
                 dim_feedforward: int = 3072, dropout: float = 0.1, activation=GELU, layer_norm_eps: float = 1e-5):
        """
        Reaction TransformerDecoder layer.

        :param max_neighbors: maximum atoms neighbors count.
        :param max_distance: maximal distance between atoms.
        :param num_in_layers: intramolecular layers count
        :param num_ex_layers: reaction-level layers count
        :param shared_in_weights: ALBERT-like intramolecular encoder layer sharing.
        :param shared_ex_weights: ALBERT-like reaction-level encoder layer sharing.
        :param shared_layers: Share MHA and FF between MoleculeEncoder's EncoderLayer and DecoderLayer.
            Self and target MHA also will share the same weights.
        """
        if shared_layers:
            assert shared_in_weights == shared_ex_weights, 'use equal weights sharing mode'
            assert n_in_head == n_ex_head, 'use equal heads count'
            if not shared_ex_weights:
                assert num_in_layers == num_ex_layers, 'use equal number of layers with' \
                                                       ' shared_ex_weights=False and shared_layers=True'
        super().__init__()
        self.molecule_encoder = MoleculeEncoder(max_neighbors=max_neighbors, max_distance=max_distance, d_model=d_model,
                                                nhead=n_in_head, num_layers=num_in_layers,
                                                dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                                                layer_norm_eps=layer_norm_eps, shared_weights=shared_in_weights)

        if shared_layers:
            self.layers = layers = []
            for encoder in ([self.molecule_encoder.layer] * num_ex_layers  # repeat shared internal layer
                            if shared_ex_weights else
                            self.molecule_encoder.layers):
                layer = DecoderLayer.__new__(DecoderLayer)
                super(Module, layer).__init__()
                layer.self_attn = encoder.self_attn
                layer.tgt_attn = encoder.self_attn
                layer.linear1 = encoder.linear1
                layer.linear2 = encoder.linear2
                layer.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
                layer.norm2 = encoder.norm1
                layer.norm3 = encoder.norm2
                layer.dropout1 = Dropout(dropout)
                layer.dropout2 = encoder.dropout1
                layer.dropout3 = encoder.dropout2
                layer.dropout4 = encoder.dropout3
                layer.activation = encoder.activation
                layers.append(layer)
        elif shared_ex_weights:
            self.layer = DecoderLayer(d_model, n_ex_head, dim_feedforward, dropout, activation, layer_norm_eps)
            self.layers = [self.layer] * num_ex_layers
        else:
            self.layers = ModuleList(DecoderLayer(d_model, n_ex_head, dim_feedforward, dropout, activation,
                                                  layer_norm_eps) for _ in range(num_ex_layers))
        self.nhead = n_ex_head

    @property
    def max_distance(self):
        """
        Distance cutoff in spatial encoder.
        """
        return self.molecule_encoder.max_distance

    def forward(self, batch: Tuple[TensorType['batch*2', 'atoms', int],
                                   TensorType['batch*2', 'atoms', int],
                                   TensorType['batch*2', 'atoms', 'atoms', int],
                                   TensorType['batch', 'atoms', float],
                                   TensorType['batch', 'atoms', float]],
                /, *, need_embedding: bool = True, need_weights: bool = False, averaged_weights: bool = False) -> \
            Union[TensorType['batch', 'atoms', 'embedding'],
                  TensorType['batch', 'atoms', 'atoms'],
                  Tuple[TensorType['batch', 'atoms', 'embedding'],
                        TensorType['batch', 'atoms', 'atoms']]]:
        """
        Atoms, Neighbors, Distances - store merged reactants in odd and products molecules in even indices of batch.
        Distances - same as molecular encoder distances but batched diagonally.
         Used 0 for disabling sharing between molecules.
        d_mask - reactants padding mask.

        :param batch: input reactions
        :param need_embedding: return atoms embeddings
        :param need_weights: return attention weights
        :param averaged_weights: return averaged attentions from each layer, otherwise only last layer
        """
        if not need_weights:
            assert not averaged_weights, 'averaging without need_weights'
            assert need_embedding, 'at least weights or embeddings should be returned'

        atoms, neighbors, distances, r_mask, p_mask = batch

        n = atoms.size(1)
        # BxN > Bx1x1xN > BxHxNxN > B*HxNxN
        r_mask = r_mask.view(-1, 1, 1, n).expand(-1, self.nhead, n, -1).flatten(end_dim=1)
        p_mask = p_mask.view(-1, 1, 1, n).expand(-1, self.nhead, n, -1).flatten(end_dim=1)

        x = self.molecule_encoder((atoms, neighbors, distances))
        rct = x[::2]  # reactants
        x = x[1::2]  # products

        if averaged_weights:  # average attention weights from each layer
            w = []
            x, a = self.layers[0](x, rct, p_mask, r_mask, disable_self_attention=True, need_weights=True)
            w.append(a)
            for lr in self.layers[1:-1]:  # noqa
                x, a = lr(x, rct, p_mask, r_mask, need_weights=True)
                w.append(a)
            x, a = self.layers[-1](x, rct, p_mask, r_mask, need_embedding=need_embedding, need_weights=True)
            w.append(a)
            w = stack(w, dim=-1).mean(-1)
            if need_embedding:
                return x, w
            return w

        # skip SA on the first layer for products
        x, _ = self.layers[0](x, rct, p_mask, r_mask, disable_self_attention=True)
        for lr in self.layers[1:-1]:  # noqa
            x, _ = lr(x, rct, p_mask, r_mask)
        x, a = self.layers[-1](x, rct, p_mask, r_mask, need_embedding=need_embedding, need_weights=need_weights)
        if need_embedding:
            if need_weights:
                return x, a
            return x
        return a


__all__ = ['ReactionDecoder']
