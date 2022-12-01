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
from torch import zeros_like, float as t_float, stack
from torch.nn import Embedding, GELU, Module, ModuleList
from torchtyping import TensorType
from typing import Tuple, Union
from ..molecule import MoleculeEncoder
from ..transformer import EncoderLayer


class ReactionEncoder(Module):
    def __init__(self, max_neighbors: int = 14, max_distance: int = 10, d_model: int = 1024, n_in_head: int = 16,
                 n_ex_head: int = 4, shared_in_weights: bool = True, shared_ex_weights: bool = True,
                 shared_layers: bool = False, num_in_layers: int = 8, num_ex_layers: int = 8,
                 dim_feedforward: int = 3072, dropout: float = 0.1, activation=GELU, layer_norm_eps: float = 1e-5):
        """
        Reaction TransformerEncoder layer.

        :param max_neighbors: maximum atoms neighbors count.
        :param max_distance: maximal distance between atoms.
        :param num_in_layers: intramolecular layers count
        :param num_ex_layers: reaction-level layers count
        :param shared_in_weights: ALBERT-like intramolecular encoder layer sharing.
        :param shared_ex_weights: ALBERT-like reaction-level encoder layer sharing.
        :param shared_layers: Use the same encoder in molecule and reaction parts.
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
        self.role_encoder = Embedding(4, d_model, 0)

        if shared_layers:
            if shared_ex_weights:  # mol encoder already shared
                self.layers = [self.molecule_encoder.layer] * num_ex_layers
            else:  # hide ModuleList from parameters lookup
                self.layers = list(self.molecule_encoder.layers)
        elif shared_ex_weights:
            self.layer = EncoderLayer(d_model, n_ex_head, dim_feedforward, dropout, activation, layer_norm_eps)
            self.layers = [self.layer] * num_ex_layers
        else:
            self.layers = ModuleList(EncoderLayer(d_model, n_ex_head, dim_feedforward, dropout, activation,
                                                  layer_norm_eps) for _ in range(num_ex_layers))
        self.nhead = n_ex_head

    @property
    def max_distance(self):
        """
        Distance cutoff in spatial encoder.
        """
        return self.molecule_encoder.max_distance

    def forward(self, batch: Tuple[TensorType['batch', 'atoms', int],
                                   TensorType['batch', 'atoms', int],
                                   TensorType['batch', 'atoms', 'atoms', int],
                                   TensorType['batch', 'atoms', int]],
                /, *, need_embedding: bool = True, need_weights: bool = False, averaged_weights: bool = False) -> \
            Union[TensorType['batch', 'atoms', 'embedding'],
                  TensorType['batch', 'atoms', 'atoms'],
                  Tuple[TensorType['batch', 'atoms', 'embedding'],
                        TensorType['batch', 'atoms', 'atoms']]]:
        """
        Use 0 for padding. Roles should be coded by 2 for reactants, 3 for products and 1 for special cls token.
        Distances - same as molecular encoder distances but batched diagonally.
         Used 0 for disabling sharing between molecules.

        :param batch: input reactions
        :param need_embedding: return atoms embeddings
        :param need_weights: return attention weights
        :param averaged_weights: return averaged attentions from each layer, otherwise only last layer
        """
        if not need_weights:
            assert not averaged_weights, 'averaging without need_weights'
            assert need_embedding, 'at least weights or embeddings should be returned'

        atoms, neighbors, distances, roles = batch
        n = atoms.size(1)
        d_mask = zeros_like(roles, dtype=t_float).masked_fill_(roles == 0, -inf).view(-1, 1, 1, n)  # BxN > Bx1x1xN >
        d_mask = d_mask.expand(-1, self.nhead, n, -1).flatten(end_dim=1)  # BxHxNxN > B*HxNxN

        # role is bert sentence encoder used to separate reactants from products and rxn CLS token coding.
        # multiplication by roles > 1 used to zeroing rxn cls token and padding. this zeroing gradients too.
        x = self.molecule_encoder((atoms, neighbors, distances)) * (roles > 1).unsqueeze(-1)
        x = x + self.role_encoder(roles)

        if averaged_weights:  # average attention weights from each layer
            w = []
            for lr in self.layers[:-1]:  # noqa
                x, a = lr(x, d_mask, need_weights=True)
                w.append(a)
            x, a = self.layers[-1](x, d_mask, need_embedding=need_embedding, need_weights=True)
            w.append(a)
            w = stack(w, dim=-1).mean(-1)
            if need_embedding:
                return x, w
            return w
        for lr in self.layers[:-1]:  # noqa
            x, _ = lr(x, d_mask)
        x, a = self.layers[-1](x, d_mask, need_embedding=need_embedding, need_weights=need_weights)
        if need_embedding:
            if need_weights:
                return x, a
            return x
        return a


__all__ = ['ReactionEncoder']
