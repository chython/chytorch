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
from torch import bmm, no_grad, Tensor
from torch.nn import Dropout, GELU, LayerNorm, LazyLinear, Linear, Module
from torch.nn.functional import cross_entropy, softmax
from torchtyping import TensorType
from typing import Optional, Union
from ._kfold import k_fold_mask


class VotingClassifier(Module):
    """
    Simple two-layer perceptron with layer normalization and dropout adopted for effective ensemble classification.
    """
    def __init__(self, ensemble: int = 10, output: int = 1, n_classes: int = 2, hidden: int = 256,
                 dropout: float = .5, activation=GELU, layer_norm_eps: float = 1e-5, loss_function=cross_entropy):
        """
        :param ensemble: number of predictive heads per output
        :param output: number of predicted properties in multitask mode. By-default single task mode is active.
        :param n_classes: number of classes
        """
        assert n_classes >= 2, 'number of classes should be higher or equal than 2'
        assert ensemble > 0 and output > 0, 'ensemble and output should be positive integers'
        super().__init__()
        self.linear1 = LazyLinear(hidden * ensemble * output)
        self.layer_norm = LayerNorm(hidden, layer_norm_eps)
        self.activation = activation()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(hidden, ensemble * output * n_classes)
        self.loss_function = loss_function

        self._n_classes = n_classes
        self._ensemble = ensemble
        self._hidden = hidden
        self._output = output

    def forward(self, x):
        """
        Returns ensemble of predictions in shape [Batch x Ensemble x Classes].
        """
        # B x E >> B x N*H >> B x N x H >> N x B x H
        x = self.linear1(x).view(-1, self._ensemble * self._output, self._hidden).transpose(0, 1)
        x = self.dropout(self.activation(self.layer_norm(x)))
        # N * C x H >> N x C x H >> N x H x C
        w = self.linear2.weight.view(-1, self._n_classes, self._hidden).transpose(1, 2)
        # N x B x C >> B x N x C
        x = bmm(x, w).transpose(0, 1).contiguous() + self.linear2.bias.view(-1, self._n_classes)
        if self._output != 1:  # MT mode
            return x.view(-1, self._output, self._ensemble, self._n_classes)  # B x O x E x C
        return x  # B x E x C

    def loss(self, x: TensorType['batch', 'embedding'],
             y: Union[TensorType['batch', 1, int], TensorType['batch', 'output', int]],
             k_fold: Optional[int] = None, ignore_index: int = -100) -> Tensor:
        """
        Apply loss function to ensemble of predictions.

        Note: y should be a column vector in single task or Batch x Output matrix in multitask mode.

        :param x: features
        :param y: properties
        :param k_fold: Cross-validation training procedure of ensemble model.
            Only k - 1 / k heads of ensemble trains on k - 1 / k items of each batch.
            On validation step 1 / k of same batches used to evaluate heads.
            Batch and ensemble sizes should be divisible by k. Disabled by default.
        """
        p = self.forward(x)  # B x E x C or B x O x E x C
        if self._output != 1:  # MT mode
            y = y.unsqueeze(-1).expand(-1, -1, self._ensemble)  # B x O > B x O x 1 > B x O x E
        else:
            y = y.expand(-1, self._ensemble)  # B x E

        if k_fold is not None:
            m = k_fold_mask(k_fold, self._ensemble, x.size(0), not self.training, p.device).bool()  # B x E
            if self._output != 1:
                # B x E > B x 1 x E
                m = m.unsqueeze(1)
            y = y.masked_fill(m, ignore_index)

        # B x E x C >> B * E x C
        # B x O x E x C >> B * O * E x C
        p = p.flatten(end_dim=-2)
        # B x E >> B * E
        # B x O x E >>  B * O * E
        y = y.flatten()
        return self.loss_function(p, y)

    @no_grad()
    def predict(self, x: TensorType['batch', 'embedding']) -> Union[TensorType['batch', int],
                                                                    TensorType['batch', 'output', int]]:
        """
        Average class prediction
        """
        return self.predict_proba(x).argmax(-1)  # B or B x O

    @no_grad()
    def predict_proba(self, x: TensorType['batch', 'embedding'], *,
                      return_std: bool = False) -> Union[TensorType['batch', 'classes', float],
                                                         TensorType['batch', 'output', 'classes', float]]:
        """
        Average probability

        :param x: features
        :param return_std: return average probability and ensemble standard deviation.
        """
        p = softmax(self.forward(x), -1)
        if return_std:
            return p.mean(-2), p.std(-2)
        return p.mean(-2)  # B x C or B x O x C


__all__ = ['VotingClassifier']
