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
from torch import zeros_like, float as t_float, inf
from torch.nn import Embedding, GELU, Module
from torchtyping import TensorType
from typing import Tuple, Union
from .molecule import MoleculeEncoder
from .transformer import EncoderLayer


class ReactionEncoder(Module):
    def __init__(self, max_neighbors: int = 14, max_distance: int = 10, d_model: int = 1024,
                 n_in_head: int = 16, n_ex_head: int = 4, num_in_layers: int = 8, num_ex_layers: int = 8,
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
        self.nhead = n_ex_head
        self.num_layers = num_ex_layers

    def forward(self, atoms: TensorType['batch', 'tokens', int], neighbors: TensorType['batch', 'tokens', int],
                distances: TensorType['batch', 'tokens', 'tokens', int], roles: TensorType['batch', 'tokens', int], *,
                need_embedding: bool = True, need_weights: bool = False) -> \
            Union[TensorType['batch', 'tokens', 'embedding'],
                  TensorType['batch', 'tokens', 'tokens'],
                  Tuple[TensorType['batch', 'tokens', 'embedding'],
                        TensorType['batch', 'tokens', 'tokens']]]:
        """
        Use 0 for padding. Roles should be coded by 2 for reactants, 3 for products and 1 for special cls token.
        Distances - same as molecular encoder distances but batched diagonally.
         Used 0 for disabling sharing between molecules.
        """
        assert need_weights or need_embedding, 'at least weights or embeddings should be returned'

        n = atoms.size(1)
        p_mask: TensorType['batch', 'tokens', float] = zeros_like(roles, dtype=t_float)
        p_mask.masked_fill_(roles == 0, -inf)
        p_mask: TensorType['batch*heads', 'tokens', 'tokens'] = p_mask.view(-1, 1, 1, n).\
            expand(-1, self.nhead, n, -1).reshape(-1, n, n)

        # role is bert sentence encoder used to separate reactants from products and rxn CLS token coding.
        # multiplication by roles > 1 used to zeroing rxn and mol cls tokens and/or padding
        x = self.molecule_encoder(atoms, neighbors, distances) * (roles > 1).unsqueeze(-1) + self.role_encoder(roles)
        for _ in range(1, self.num_layers):
            x, _ = self.layer(x, p_mask)
        x, a = self.layer(x, p_mask, need_embedding=need_embedding, need_weights=need_weights)
        if need_embedding:
            if need_weights:
                return x, a
            return x
        return a


__all__ = ['ReactionEncoder']
