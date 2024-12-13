# -*- coding: utf-8 -*-
#
# Copyright 2021-2024 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from math import inf
from torch import zeros_like, float as t_float
from torch.nn import Embedding, GELU, Module
from torchtyping import TensorType
from .molecule import MoleculeEncoder
from .transformer import EncoderLayer
from ..utils.data import ReactionEncoderDataBatch


class ReactionEncoder(Module):
    def __init__(self, max_neighbors: int = 14, max_distance: int = 10, d_model: int = 1024, n_in_head: int = 16,
                 n_ex_head: int = 4, num_in_layers: int = 8, num_ex_layers: int = 8,
                 dim_feedforward: int = 3072, dropout: float = 0.1, activation=GELU, layer_norm_eps: float = 1e-5):
        """
        Reaction Graphormer from https://doi.org/10.1021/acs.jcim.2c00344.

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
        d_mask = d_mask.expand(-1, self.nhead, n, -1)  # > BxHxNxN

        # role is bert sentence encoder used to separate reactants from products and rxn CLS token coding.
        # multiplication by roles > 1 used to zeroing rxn cls token and padding. this zeroing gradients too.
        x = self.molecule_encoder((atoms, neighbors, distances)) * (roles > 1).unsqueeze_(-1)
        x = x + self.role_encoder(roles)

        for lr in self.layers:
            x, _ = lr(x, d_mask)
        return x


__all__ = ['ReactionEncoder']
