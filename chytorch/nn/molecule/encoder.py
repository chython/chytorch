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
from itertools import repeat
from torch.nn import GELU, Module, ModuleList, LayerNorm
from torchtyping import TensorType
from typing import Tuple, Optional, List
from warnings import warn
from ._embedding import EmbeddingBag
from ..lora import Embedding
from ..transformer import EncoderLayer
from ...utils.data import MoleculeDataBatch


def _update(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'centrality_encoder.weight' in state_dict:
        warn('fixed chytorch<1.37 checkpoint', DeprecationWarning)
        state_dict[prefix + 'neighbors_encoder.weight'] = state_dict.pop(prefix + 'centrality_encoder.weight')
        state_dict[prefix + 'distance_encoder.weight'] = state_dict.pop(prefix + 'spatial_encoder.weight')
    if prefix + 'atoms_encoder.weight' in state_dict:
        warn('fixed chytorch<1.61 checkpoint', DeprecationWarning)
        state_dict[prefix + 'embedding.atoms_encoder.weight'] = state_dict.pop(prefix + 'atoms_encoder.weight')
        state_dict[prefix + 'embedding.neighbors_encoder.weight'] = state_dict.pop(prefix + 'neighbors_encoder.weight')


class MoleculeEncoder(Module):
    """
    Inspired by https://arxiv.org/pdf/2106.05234.pdf
    """
    def __init__(self, max_neighbors: int = 14, max_distance: int = 10, d_model: int = 1024, nhead: int = 16,
                 num_layers: int = 8, dim_feedforward: int = 3072, shared_weights: bool = True,
                 shared_attention_bias: bool = True, dropout: float = 0.1, activation=GELU,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False, post_norm: bool = False,
                 zero_bias: bool = False, perturbation: float = 0., max_tokens: int = 121,
                 projection_bias: bool = True, ff_bias: bool = True):
        """
        Molecule Graphormer from https://doi.org/10.1021/acs.jcim.2c00344.

        :param max_neighbors: maximum atoms neighbors count.
        :param max_distance: maximal distance between atoms.
        :param shared_weights: ALBERT-like encoder weights sharing.
        :param norm_first: do pre-normalization in encoder layers.
        :param post_norm: do normalization of output. Works only when norm_first=True.
        :param zero_bias: use frozen zero bias of attention for non-reachable atoms.
        :param perturbation: add perturbation to embedding (https://aclanthology.org/2021.naacl-main.460.pdf).
            Disabled by default
        :param shared_attention_bias: use shared distance encoder or unique for each transformer layer.
        :param max_tokens: number of tokens in the atom encoder embedding layer.
        """
        super().__init__()
        self.embedding = EmbeddingBag(max_neighbors, d_model, perturbation, max_tokens)

        self.shared_attention_bias = shared_attention_bias
        if shared_attention_bias:
            self.distance_encoder = Embedding(max_distance + 3, nhead, int(zero_bias) or None, neg_inf_idx=0)
            # None filled encoders mean reusing previously calculated bias. possible manually create different arch.
            # this done for speedup in comparison to layer duplication.
            self.distance_encoders = [None] * num_layers
            self.distance_encoders[0] = self.distance_encoder  # noqa
        else:
            self.distance_encoders = ModuleList(Embedding(max_distance + 3, nhead,
                                                          int(zero_bias) or None, neg_inf_idx=0)
                                                for _ in range(num_layers))

        self.max_distance = max_distance
        self.max_neighbors = max_neighbors
        self.perturbation = perturbation
        self.num_layers = num_layers
        self.max_tokens = max_tokens
        self.post_norm = post_norm
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.norm_first = norm_first
        self.zero_bias = zero_bias
        if post_norm:
            assert norm_first, 'post_norm requires norm_first'
            self.norm = LayerNorm(d_model, layer_norm_eps)

        self.shared_weights = shared_weights
        if shared_weights:
            self.layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first,
                                      projection_bias=projection_bias, ff_bias=ff_bias)
            self.layers = [self.layer] * num_layers
        else:
            # layers sharing scheme can be manually changed. e.g. pairs of shared encoders
            self.layers = ModuleList(EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                  layer_norm_eps, norm_first, projection_bias=projection_bias,
                                                  ff_bias=ff_bias) for _ in range(num_layers))
        self._register_load_state_dict_pre_hook(_update)

    def forward(self, batch: MoleculeDataBatch, /, *,
                cache: Optional[List[Tuple[TensorType['batch', 'atoms+conditions', 'embedding'],
                                           TensorType['batch', 'atoms+conditions', 'embedding']]]] = None) -> \
            TensorType['batch', 'atoms', 'embedding']:
        """
        Use 0 for padding.
        Atoms should be coded by atomic numbers + 2.
        Token 1 reserved for cls token, 2 reserved for molecule cls or training tricks like MLM.
        Neighbors should be coded from 2 (means no neighbors) to max neighbors + 2.
        Neighbors equal to 1 reserved for training tricks like MLM. Use 0 for cls.
        Distances should be coded from 2 (means self-loop) to max_distance + 2.
        Non-reachable atoms should be coded by 1.
        """
        cache = repeat(None) if cache is None else iter(cache)
        atoms, neighbors, distances = batch

        x = self.embedding(atoms, neighbors)

        for lr, d, c in zip(self.layers, self.distance_encoders, cache):
            if d is not None:
                d_mask = d(distances).permute(0, 3, 1, 2)  # BxNxNxH > BxHxNxN
            # else: reuse previously calculated mask
            x, _ = lr(x, d_mask, cache=c)  # noqa

        if self.post_norm:
            return self.norm(x)
        return x

    @property
    def centrality_encoder(self):
        warn('centrality_encoder renamed to neighbors_encoder in chytorch 1.37', DeprecationWarning)
        return self.neighbors_encoder

    @property
    def spatial_encoder(self):
        warn('spatial_encoder renamed to distance_encoder in chytorch 1.37', DeprecationWarning)
        return self.distance_encoder

    @property
    def atoms_encoder(self):
        warn('neighbors_encoder moved to embedding submodule in chytorch 1.61', DeprecationWarning)
        return self.embedding.atoms_encoder

    @property
    def neighbors_encoder(self):
        warn('neighbors_encoder moved to embedding submodule in chytorch 1.61', DeprecationWarning)
        return self.embedding.neighbors_encoder


__all__ = ['MoleculeEncoder']
