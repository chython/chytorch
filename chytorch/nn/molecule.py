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
from torch import no_grad, stack, ones, long, empty_like
from torch.nn import Embedding, GELU, Module, ModuleList, LayerNorm
from torchtyping import TensorType
from typing import Tuple, Union, List
from warnings import warn
from .transformer import EncoderLayer
from ..utils.data import MoleculeDataBatch


def _hook(x):
    return x.index_fill(0, ones(1, dtype=long, device=x.device), 0)


def _update(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'centrality_encoder.weight' in state_dict:
        warn('fixed chytorch<1.37 checkpoint', DeprecationWarning)
        state_dict[prefix + 'neighbors_encoder.weight'] = state_dict.pop(prefix + 'centrality_encoder.weight')
        state_dict[prefix + 'distance_encoder.weight'] = state_dict.pop(prefix + 'spatial_encoder.weight')


class MoleculeEncoder(Module):
    """
    Inspired by https://arxiv.org/pdf/2106.05234.pdf
    """
    def __init__(self, max_neighbors: int = 14, max_distance: int = 10,  max_tokens: int = 0,
                 d_model: int = 1024, nhead: int = 16, num_layers: int = 8, dim_feedforward: int = 3072,
                 shared_weights: bool = True, dropout: float = 0.1, activation=GELU, layer_norm_eps: float = 1e-5,
                 norm_first: bool = False, post_norm: bool = False, zero_bias: bool = False, perturbation: float = 0.,
                 positional_distance: int = 0):
        """
        Molecule TransformerEncoder layer.

        :param max_neighbors: maximum atoms neighbors count.
        :param max_distance: maximal distance between atoms.
        :param max_tokens: number of non-atomic tokens.
        :param shared_weights: ALBERT-like encoder weights sharing.
        :param norm_first: do pre-normalization in encoder layers.
        :param post_norm: do normalization of output. Works only when norm_first=True.
        :param zero_bias: use frozen zero bias of attention for non-reachable atoms.
        :param perturbation: add perturbation to embedding (https://aclanthology.org/2021.naacl-main.460.pdf).
            Disabled by default
        :param positional_distance: ALIBI-like (but learnable) positional encoding threshold. Disabled by default.
        """
        assert perturbation >= 0, 'zero or positive perturbation expected'
        super().__init__()
        if positional_distance:
            assert positional_distance > 1, 'positional distance should be greater than 1 or disabled'
            self.positional_distance = positional_distance
            positional_distance -= 1
        else:
            self.positional_distance = 0
        self.atoms_encoder = Embedding(121 + (max_tokens and max_tokens + 2), d_model, 0)
        self.neighbors_encoder = Embedding(max_neighbors + 3, d_model, 0)
        self.distance_encoder = Embedding(positional_distance + max_distance + 3, nhead, 0)

        self.max_distance = max_distance
        self.max_tokens = max_tokens
        self.max_neighbors = max_neighbors
        self.perturbation = perturbation
        self.post_norm = post_norm
        if post_norm:
            assert norm_first, 'post_norm requires norm_first'
            self.norm = LayerNorm(d_model, layer_norm_eps)

        if shared_weights:
            self.layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first)
            self.layers = [self.layer] * num_layers
        else:
            self.layers = ModuleList(EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                  layer_norm_eps, norm_first) for _ in range(num_layers))

        with no_grad():  # trick to disable padding attention
            self.distance_encoder.weight[0].fill_(-inf)
            if zero_bias:
                self.distance_encoder.weight[1].fill_(0)
                self.distance_encoder.weight.register_hook(_hook)
        self._register_load_state_dict_pre_hook(_update)

    def forward(self, batch: MoleculeDataBatch, /, *, need_embedding: bool = True, need_weights: bool = False,
                averaged_weights: bool = False, intermediate_embeddings: bool = False) -> \
            Union[TensorType['batch', 'atoms', 'embedding'], TensorType['batch', 'atoms', 'atoms'],
                  Tuple[TensorType['batch', 'atoms', 'embedding'], TensorType['batch', 'atoms', 'atoms']],
                  Tuple[TensorType['batch', 'atoms', 'embedding'], List[TensorType['batch', 'atoms', 'embedding']]],
                  Tuple[TensorType['batch', 'atoms', 'embedding'],
                        TensorType['batch', 'atoms', 'atoms'],
                        List[TensorType['batch', 'atoms', 'embedding']]]]:
        """
        Use 0 for padding.
        Atoms should be coded by atomic numbers + 2.
        Token 1 reserved for cls token, 2 reserved for reaction cls or training tricks like MLM.
        Neighbors should be coded from 2 (means no neighbors) to max neighbors + 2.
        Neighbors equal to 1 reserved for training tricks like MLM. Use 0 for cls.
        Distances should be coded from 2 (means self-loop) to max_distance + 2.
        Non-reachable atoms should be coded by 1.

        :param need_embedding: return atoms embeddings
        :param need_weights: return attention weights
        :param averaged_weights: return averaged attentions from each layer, otherwise only last layer
        :param intermediate_embeddings: return embedding of each layer including initial weights but last
        """
        if not need_weights:
            assert not averaged_weights, 'averaging without need_weights'
            assert need_embedding, 'at least weights or embeddings should be returned'
        elif intermediate_embeddings:
            assert need_embedding, 'need_embedding should be active for intermediate_embeddings option'

        atoms, neighbors, distances = batch
        d_mask = self.distance_encoder(distances).permute(0, 3, 1, 2).flatten(end_dim=1)  # BxNxNxH > BxHxNxN > B*HxNxN

        # cls token in neighbors coded by 0
        x = self.atoms_encoder(atoms) + self.neighbors_encoder(neighbors)

        if self.perturbation and self.training:
            x = x + empty_like(x).uniform_(-self.perturbation, self.perturbation)

        if intermediate_embeddings:
            embeddings = [x]

        if averaged_weights:  # average attention weights from each layer
            w = []
            for lr in self.layers[:-1]:  # noqa
                x, a = lr(x, d_mask, need_weights=True)
                w.append(a)
                if intermediate_embeddings:
                    embeddings.append(x)  # noqa
            x, a = self.layers[-1](x, d_mask, need_embedding=need_embedding, need_weights=True)
            w.append(a)
            w = stack(w, dim=-1).mean(-1)
            if need_embedding:
                if self.post_norm:
                    x = self.norm(x)
                if intermediate_embeddings:
                    return x, w, embeddings
                return x, w
            return w

        for lr in self.layers[:-1]:  # noqa
            x, _ = lr(x, d_mask)
            if intermediate_embeddings:
                embeddings.append(x)  # noqa
        x, a = self.layers[-1](x, d_mask, need_embedding=need_embedding, need_weights=need_weights)
        if need_embedding:
            if self.post_norm:
                x = self.norm(x)
            if intermediate_embeddings:
                if need_weights:
                    return x, a, embeddings
                return x, embeddings
            elif need_weights:
                return x, a
            return x
        return a

    @property
    def centrality_encoder(self):
        warn('centrality_encoder renamed to neighbors_encoder in chytorch 1.37', DeprecationWarning)
        return self.neighbors_encoder

    @property
    def spatial_encoder(self):
        warn('spatial_encoder renamed to distance_encoder in chytorch 1.37', DeprecationWarning)
        return self.distance_encoder


__all__ = ['MoleculeEncoder']
