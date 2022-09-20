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
from functools import lru_cache
from torch import bmm, no_grad, sigmoid, ones, bool as t_bool
from torch.nn import Dropout, GELU, LayerNorm, LazyLinear, Linear, Module
from torch.nn.functional import cross_entropy, l1_loss, softmax, binary_cross_entropy_with_logits
from typing import Optional


@lru_cache()
def k_fold_mask(k_fold, ensemble, batch_size, device=None):
    assert k_fold >= 3, 'k-fold should be at least 3'
    assert not ensemble % k_fold, 'ensemble should be divisible by k-fold'
    assert not batch_size % k_fold, 'batch size should be divisible by k-fold'

    m = ones(batch_size, ensemble, dtype=t_bool, device=device)  # k-th fold mask
    batch_size //= k_fold
    ensemble //= k_fold
    for n in range(k_fold):  # disable folds
        m[n * batch_size: n * batch_size + batch_size, n * ensemble: n * ensemble + ensemble] = False
    return m


class VotingRegressor(Module):
    """
    Simple two-layer perceptron with layer normalization and dropout adopted for effective ensemble regression modeling.
    """
    def __init__(self, ensemble: int = 10, output: int = 1, hidden: int = 256, dropout: float = .5,
                 activation=GELU, layer_norm_eps: float = 1e-5, loss_function=l1_loss):
        """
        :param ensemble: number of predictive heads per output
        :param output: number of predicted properties in multitask mode. By-default single task mode is active.
        """
        assert ensemble > 0 and output > 0, 'ensemble and output should be positive integers'
        super().__init__()
        self.linear1 = LazyLinear(hidden * ensemble * output)
        self.layer_norm = LayerNorm(hidden, layer_norm_eps)
        self.activation = activation()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(hidden, ensemble * output)
        self.loss_function = loss_function

        self._ensemble = ensemble
        self._hidden = hidden
        self._output = output

    def forward(self, x):
        """
        Returns ensemble of predictions in shape [Batch x Output*Ensemble].
        """
        # B x E >> B x N*H >> B x N x H >> N x B x H
        x = self.linear1(x).view(-1, self._ensemble * self._output, self._hidden).transpose(0, 1)
        x = self.dropout(self.activation(self.layer_norm(x)))
        # N x H >> N x H x 1
        w = self.linear2.weight.unsqueeze(2)
        # N x B x 1 >> N x B >> B x N
        return bmm(x, w).squeeze(-1).transpose(0, 1).contiguous() + self.linear2.bias

    def loss(self, x, y, k_fold: Optional[int] = None):
        """
        Apply loss function to ensemble of predictions.

        Note: y should be a column vector in single task or Batch x Output matrix in multitask mode.

        :param k_fold: Cross-validation training procedure of ensemble model.
            Only k - 1 / k heads of ensemble trains on k - 1 / k items of each batch.
            On validation step 1 / k of same batches used to evaluate heads.
            Batch and ensemble sizes should be divisible by k. Disabled by default.
        """
        p = self.forward(x)
        if self._output != 1:  # MT mode
            p = p.view(-1, self._output, self._ensemble).transpose(1, 2)  # B x O x E > B x E x O
            y = y.unsqueeze(1)  # B x O > B x 1 x O
        y = y.expand(p.size())  # B x E or B x E x O

        if k_fold is not None:
            m = k_fold_mask(k_fold, self._ensemble, x.size(0), p.device)
            if not self.training:  # validation/test mode
                m = ~m
            p = p[m]
            y = y[m]
        return self.loss_function(p, y)

    @no_grad()
    def predict(self, x, *, return_std: bool = False):
        """
        Average prediction

        :param return_std: return average prediction and ensemble standard deviation.
        """
        p = self.forward(x)  # B x N
        if self._output != 1:  # MT mode
            p = p.view(-1, self._output, self._ensemble)  # B x O x E
        if return_std:
            return p.mean(-1), p.std(-1)
        return p.mean(-1)


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
        return bmm(x, w).transpose(0, 1).contiguous() + self.linear2.bias.view(-1, self._n_classes)

    def loss(self, x, y, k_fold: Optional[int] = None):
        """
        Apply loss function to ensemble of predictions.

        Note: y should be a column vector in single task or Batch x Output matrix in multitask mode.

        :param k_fold: Cross-validation training procedure of ensemble model.
            Only k - 1 / k heads of ensemble trains on k - 1 / k items of each batch.
            On validation step 1 / k of same batches used to evaluate heads.
            Batch and ensemble sizes should be divisible by k. Disabled by default.
        """
        p = self.forward(x)  # B x E x C
        if self._output != 1:  # MT mode
            # B x O * E x C > B x O x E x C > B x E x O x C
            p = p.view(-1, self._output, self._ensemble, self._n_classes).transpose(1, 2)
            y = y.unsqueeze(1).expand(-1, self._ensemble, -1)  # B x O > B x 1 x O > B x E x O
        else:
            y = y.expand(-1, self._ensemble)  # B x E

        if k_fold is not None:
            m = k_fold_mask(k_fold, self._ensemble, x.size(0), p.device)  # B x E
            if not self.training:  # validation/test mode
                m = ~m
            p = p[m]  # M x C or M x O x C
            y = y[m]  # M or M x O

        # B x E x C >> B * E x C
        # B x E x O x C >> B * E * O x C
        # M x C >> M x C
        # M x O x C >> M * O x C
        p = p.flatten(end_dim=-2)
        # B x E >> B * E
        # B x E x O >>  B * E * O
        # M >> M
        # M x O >> M * O
        y = y.flatten()
        return self.loss_function(p, y)

    @no_grad()
    def predict(self, x):
        """
        Average class prediction
        """
        return self.predict_proba(x).argmax(-1)

    @no_grad()
    def predict_proba(self, x, return_std: bool = False):
        """
        Average probability

        :param return_std: return average probability and ensemble standard deviation.
        """
        p = softmax(self.forward(x), -1)
        if self._output != 1:  # MT mode
            p = p.view(-1, self._output, self._ensemble, self._n_classes)  # B x O x E x C
        if return_std:
            return p.mean(-2), p.std(-2)
        return p.mean(-2)


class BinaryVotingClassifier(VotingRegressor):
    """
    Simple two-layer perceptron with layer normalization and dropout adopted for effective
    ensemble binary classification tasks.
    """
    def __init__(self, ensemble: int = 10, output: int = 1, hidden: int = 256, dropout: float = .5,
                 activation=GELU, layer_norm_eps: float = 1e-5, loss_function=binary_cross_entropy_with_logits):
        super().__init__(ensemble, output, hidden, dropout, activation, layer_norm_eps, loss_function)

    @no_grad()
    def predict(self, x):
        """
        Average class prediction
        """
        return (self.predict_proba(x) > .5).long()

    @no_grad()
    def predict_proba(self, x, *, return_std: bool = False):
        """
        Average probability

        :param return_std: return average probability and ensemble standard deviation.
        """
        p = sigmoid(self.forward(x))  # B x N
        if self._output != 1:  # MT mode
            p = p.view(-1, self._output, self._ensemble)  # B x O x E
        if return_std:
            return p.mean(-1), p.std(-1)
        return p.mean(-1)


__all__ = ['VotingRegressor', 'VotingClassifier', 'BinaryVotingClassifier', 'k_fold_mask']
