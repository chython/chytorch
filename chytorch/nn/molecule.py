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
from math import inf
from torch import no_grad
from torch.nn import Embedding, GELU, Module, ModuleList
from torchtyping import TensorType
from typing import Tuple, Union
from .transformer import EncoderLayer


class MoleculeEncoder(Module):
    """
    Inspired by https://arxiv.org/pdf/2106.05234.pdf
    """
    def __init__(self, *, max_neighbors: int = 14, max_distance: int = 10, shared_layers: bool = True,
                 d_model: int = 1024, nhead: int = 16, num_layers: int = 8, dim_feedforward: int = 3072,
                 dropout: float = 0.1, activation=GELU, layer_norm_eps: float = 1e-5):
        """
        Molecule TransformerEncoder layer.

        :param max_neighbors: maximum atoms neighbors count.
        :param max_distance: maximal distance between atoms.
        :param shared_layers: ALBERT-like encoder layer sharing.
        """
        super().__init__()
        self.atoms_encoder = Embedding(121, d_model, 0)
        self.centrality_encoder = Embedding(max_neighbors + 3, d_model, 0)
        self.spatial_encoder = Embedding(max_distance + 3, nhead, 0)

        if shared_layers:
            self.layer = layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
            self.layers = [layer] * num_layers
        else:
            self.layers = ModuleList(EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
                                     for _ in range(num_layers))

        with no_grad():  # trick to disable padding attention
            self.spatial_encoder.weight[0].fill_(-inf)

    @property
    def max_distance(self):
        """
        Distance cutoff in spatial encoder.
        """
        return self.spatial_encoder.num_embeddings - 3

    def forward(self, ands: Tuple[TensorType['batch', 'tokens', int],
                                  TensorType['batch', 'tokens', int],
                                  TensorType['batch', 'tokens', 'tokens', int]],
                /, *, need_embedding: bool = True, need_weights: bool = False) -> \
            Union[TensorType['batch', 'tokens', 'embedding'], TensorType['batch', 'tokens', 'tokens'],
                  Tuple[TensorType['batch', 'tokens', 'embedding'], TensorType['batch', 'tokens', 'tokens']]]:
        """
        Use 0 for padding.
        Atoms should be coded by atomic numbers + 2. 1 can be used for cls token, 2 reserved for masks in MLM.
        Neighbors should be coded from 2 (means no neighbors) to max neighbors + 2.
        Neighbors equal to 1 reserved for masks in MLM. Use 0 for cls.
        Distances should be coded from 2 (means self-loop) to max_distance + 2.
        Non-reachable atoms should be coded by 1.
        """
        assert need_weights or need_embedding, 'at least weights or embeddings should be returned'

        atoms, neighbors, distances = ands
        d_mask: TensorType['batch', 'tokens', 'tokens', 'heads'] = self.spatial_encoder(distances)
        d_mask: TensorType['batch*heads', 'tokens', 'tokens'] = d_mask.permute(0, 3, 1, 2).flatten(end_dim=1)

        # cls token in neighbors coded by 0 to disable centrality encoding.
        x = self.atoms_encoder(atoms) + self.centrality_encoder(neighbors)
        for lr in self.layers[:-1]:
            x, _ = lr(x, d_mask)
        x, a = self.layers[-1](x, d_mask, need_embedding=need_embedding, need_weights=need_weights)
        if need_embedding:
            if need_weights:
                return x, a
            return x
        return a


__all__ = ['MoleculeEncoder']
