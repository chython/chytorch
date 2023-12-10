# -*- coding: utf-8 -*-
#
# Copyright 2021-2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from torch import empty_like
from torch.nn import GELU, Module, ModuleList, LayerNorm
from torchtyping import TensorType
from warnings import warn
from ..lora import Embedding
from ..transformer import EncoderLayer
from ...utils.data import MoleculeDataBatch


def _update(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'centrality_encoder.weight' in state_dict:
        warn('fixed chytorch<1.37 checkpoint', DeprecationWarning)
        state_dict[prefix + 'neighbors_encoder.weight'] = state_dict.pop(prefix + 'centrality_encoder.weight')
        state_dict[prefix + 'distance_encoder.weight'] = state_dict.pop(prefix + 'spatial_encoder.weight')


class MoleculeEncoder(Module):
    """
    Inspired by https://arxiv.org/pdf/2106.05234.pdf
    """
    def __init__(self, max_neighbors: int = 14, max_distance: int = 10, d_model: int = 1024, nhead: int = 16,
                 num_layers: int = 8, dim_feedforward: int = 3072, shared_weights: bool = True,
                 shared_attention_bias: bool = True, dropout: float = 0.1, activation=GELU,
                 layer_norm_eps: float = 1e-5, norm_first: bool = False, post_norm: bool = False,
                 zero_bias: bool = False, perturbation: float = 0., max_tokens: int = 121,
                 lora_r: int = 0, lora_alpha: float = 1., lora_dropout: float = 0.):
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
        :param lora_r: LoRA factorization dimension size in encoder embeddings. Disabled by default.
        :param lora_alpha: LoRA scaling factor.
        :param lora_dropout: LoRA input dropout.
        :param shared_attention_bias: use shared distance encoder or unique for each transformer layer.
        :param max_tokens: number of tokens in the atom encoder embedding layer.
        """
        assert max_tokens >= 121, 'at least 121 tokens should be'
        assert perturbation >= 0, 'zero or positive perturbation expected'
        super().__init__()
        self.atoms_encoder = Embedding(max_tokens, d_model, 0, lora_r=lora_r, lora_alpha=lora_alpha)
        self.neighbors_encoder = Embedding(max_neighbors + 3, d_model, 0, lora_r=lora_r, lora_alpha=lora_alpha)

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
        if post_norm:
            assert norm_first, 'post_norm requires norm_first'
            self.norm = LayerNorm(d_model, layer_norm_eps)

        self.shared_weights = shared_weights
        if shared_weights:
            self.layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, norm_first,
                                      lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.layers = [self.layer] * num_layers
        else:
            # layers sharing scheme can be manually changed. e.g. pairs of shared encoders
            self.layers = ModuleList(EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                  layer_norm_eps, norm_first, lora_r=lora_r, lora_alpha=lora_alpha,
                                                  lora_dropout=lora_dropout) for _ in range(num_layers))
        self._register_load_state_dict_pre_hook(_update)

    def forward(self, batch: MoleculeDataBatch) -> TensorType['batch', 'atoms', 'embedding']:
        """
        Use 0 for padding.
        Atoms should be coded by atomic numbers + 2.
        Token 1 reserved for cls token, 2 reserved for molecule cls or training tricks like MLM.
        Neighbors should be coded from 2 (means no neighbors) to max neighbors + 2.
        Neighbors equal to 1 reserved for training tricks like MLM. Use 0 for cls.
        Distances should be coded from 2 (means self-loop) to max_distance + 2.
        Non-reachable atoms should be coded by 1.
        """
        atoms, neighbors, distances = batch

        # cls token in neighbors coded by 0
        x = self.atoms_encoder(atoms) + self.neighbors_encoder(neighbors)

        if self.perturbation and self.training:
            x = x + empty_like(x).uniform_(-self.perturbation, self.perturbation)

        for lr, d in zip(self.layers, self.distance_encoders):
            if d is not None:
                d_mask = d(distances).permute(0, 3, 1, 2)  # BxNxNxH > BxHxNxN
            # else: reuse previously calculated mask
            x, _ = lr(x, d_mask)  # noqa

        if self.post_norm:
            return self.norm(x)
        return x

    def merge_lora(self):
        """
        Transform LoRA layers to normal
        """
        self.atoms_encoder.merge_lora()
        self.neighbors_encoder.merge_lora()
        for layer in self.layers:
            layer.merge_lora()

    @property
    def centrality_encoder(self):
        warn('centrality_encoder renamed to neighbors_encoder in chytorch 1.37', DeprecationWarning)
        return self.neighbors_encoder

    @property
    def spatial_encoder(self):
        warn('spatial_encoder renamed to distance_encoder in chytorch 1.37', DeprecationWarning)
        return self.distance_encoder


__all__ = ['MoleculeEncoder']
