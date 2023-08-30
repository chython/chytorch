# -*- coding: utf-8 -*-
#
#  Copyright 2021-2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from torch import zeros_like, float as t_float
from torch.nn import Embedding, GELU, Module
from torchtyping import TensorType
from ..molecule import MoleculeEncoder
from ..transformer import EncoderLayer
from ...utils.data import ReactionEncoderDataBatch


class ReactionEncoder(Module):
    def __init__(self, max_neighbors: int = 14, max_distance: int = 10, d_model: int = 1024, n_in_head: int = 16,
                 n_ex_head: int = 4, num_in_layers: int = 8, num_ex_layers: int = 8,
                 dim_feedforward: int = 3072, dropout: float = 0.1, activation=GELU, layer_norm_eps: float = 1e-5):
        """
        Reaction TransformerEncoder layer.

        :param max_neighbors: maximum atoms neighbors count.
        :param max_distance: maximal distance between atoms.
        :param num_in_layers: intramolecular layers count
        :param num_ex_layers: reaction-level layers count
        """
        super().__init__()
        self.molecule_encoder = MoleculeEncoder(max_neighbors=max_neighbors, max_distance=max_distance, d_model=d_model,
                                                nhead=n_in_head, num_layers=num_in_layers,
                                                dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                                                layer_norm_eps=layer_norm_eps)
        self.role_encoder = Embedding(4, d_model, 0)
        self.layer = EncoderLayer(d_model, n_ex_head, dim_feedforward, dropout, activation, layer_norm_eps)
        self.layers = [self.layer] * num_ex_layers
        self.nhead = n_ex_head

    @property
    def max_distance(self):
        """
        Distance cutoff in spatial encoder.
        """
        return self.molecule_encoder.max_distance

    def forward(self, batch: ReactionEncoderDataBatch) -> TensorType['batch', 'atoms', 'embedding']:
        """
        Use 0 for padding. Roles should be coded by 2 for reactants, 3 for products and 1 for special cls token.
        Distances - same as molecular encoder distances but batched diagonally.
         Used 0 for disabling sharing between molecules.
        """
        atoms, neighbors, distances, roles = batch
        n = atoms.size(1)
        d_mask = zeros_like(roles, dtype=t_float).masked_fill_(roles == 0, -inf).view(-1, 1, 1, n)  # BxN > Bx1x1xN >
        d_mask = d_mask.expand(-1, self.nhead, n, -1).flatten(end_dim=1)  # BxHxNxN > B*HxNxN

        # role is bert sentence encoder used to separate reactants from products and rxn CLS token coding.
        # multiplication by roles > 1 used to zeroing rxn cls token and padding. this zeroing gradients too.
        x = self.molecule_encoder((atoms, neighbors, distances)) * (roles > 1).unsqueeze_(-1)
        x = x + self.role_encoder(roles)

        for lr in self.layers:
            x, _ = lr(x, d_mask)
        return x


__all__ = ['ReactionEncoder']
