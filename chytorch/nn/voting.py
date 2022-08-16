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
from torch import bmm, no_grad, sigmoid
from torch.nn import Dropout, GELU, LayerNorm, LazyLinear, Linear, Module
from torch.nn.functional import cross_entropy, l1_loss, softmax, binary_cross_entropy_with_logits


class VotingRegressor(Module):
    """
    Simple two-layer perceptron with layer normalization and dropout adopted for effective ensemble regression modeling.
    """
    def __init__(self, ensemble: int = 10, hidden: int = 256, dropout: float = .5,
                 activation=GELU, layer_norm_eps: float = 1e-5, loss_function=l1_loss):
        super().__init__()
        self.linear1 = LazyLinear(hidden * ensemble)
        self.layer_norm = LayerNorm(hidden, layer_norm_eps)
        self.activation = activation()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(hidden, ensemble)
        self.loss_function = loss_function

        self._ensemble = ensemble
        self._hidden = hidden

    def forward(self, x):
        """
        Returns ensemble of predictions in shape [Batch x Ensemble].
        """
        # B x E >> B x N*H >> B x N x H >> N x B x H
        x = self.linear1(x).view(-1, self._ensemble, self._hidden).transpose(0, 1)
        x = self.dropout(self.activation(self.layer_norm(x)))
        # N x H >> N x H x 1
        w = self.linear2.weight.unsqueeze(2)
        # N x B x 1 >> N x B >> B x N
        return bmm(x, w).squeeze(-1).transpose(0, 1).contiguous() + self.linear2.bias

    def loss(self, x, y):
        """
        Apply loss function to ensemble of predictions.

        Note: y should be a column vector.
        """
        p = self.forward(x)
        return self.loss_function(p, y.expand(p.size()))

    @no_grad()
    def predict(self, x, return_std: bool = False):
        """
        Average prediction

        :param return_std: return average prediction and ensemble standard deviation.
        """
        p = self.forward(x)
        if return_std:
            return p.mean(1), p.std(1)
        return p.mean(1)


class VotingClassifier(Module):
    """
    Simple two-layer perceptron with layer normalization and dropout adopted for effective ensemble classification.
    """
    def __init__(self, ensemble: int = 10, hidden: int = 256, n_classes: int = 2,
                 dropout: float = .5, activation=GELU, layer_norm_eps: float = 1e-5, loss_function=cross_entropy):
        super().__init__()
        self.linear1 = LazyLinear(hidden * ensemble)
        self.layer_norm = LayerNorm(hidden, layer_norm_eps)
        self.activation = activation()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(hidden, ensemble * n_classes)
        self.loss_function = loss_function

        self._n_classes = n_classes
        self._ensemble = ensemble
        self._hidden = hidden

    def forward(self, x):
        """
        Returns ensemble of predictions in shape [Batch x Ensemble x Classes].
        """
        # B x E >> B x N*H >> B x N x H >> N x B x H
        x = self.linear1(x).view(-1, self._ensemble, self._hidden).transpose(0, 1)
        x = self.dropout(self.activation(self.layer_norm(x)))
        # N * C x H >> N x C x H >> N x H x C
        w = self.linear2.weight.view(self._ensemble, -1, self._hidden).transpose(1, 2)
        # N x B x C >> B x N x C
        return bmm(x, w).transpose(0, 1).contiguous() + self.linear2.bias.view(self._ensemble, -1)

    def loss(self, x, y):
        """
        Apply loss function to ensemble of predictions.

        Note: y should be a long vector.
        """
        p = self.forward(x).flatten(end_dim=1)  # B x N x C >> B * N x C
        t = y.unsqueeze(-1).expand(-1, self._ensemble).flatten()  # B >> B x 1 >> B x N >> B * N
        return self.loss_function(p, t)

    @no_grad()
    def predict(self, x):
        """
        Average class prediction
        """
        return softmax(self.forward(x), 2).mean(1).argmax(1)

    @no_grad()
    def predict_proba(self, x, return_std: bool = False):
        """
        Average probability

        :param return_std: return average probability and ensemble standard deviation.
        """
        p = softmax(self.forward(x), 2)
        if return_std:
            return p.mean(1), p.std(1)
        return p.mean(1)


class BinaryVotingClassifier(Module):
    """
    Simple two-layer perceptron with layer normalization and dropout adopted for effective
    ensemble binary classification tasks.
    """
    def __init__(self, ensemble: int = 10, hidden: int = 256, dropout: float = .5,
                 activation=GELU, layer_norm_eps: float = 1e-5, loss_function=binary_cross_entropy_with_logits):
        super().__init__()
        self.linear1 = LazyLinear(hidden * ensemble)
        self.layer_norm = LayerNorm(hidden, layer_norm_eps)
        self.activation = activation()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(hidden, ensemble)
        self.loss_function = loss_function

        self._ensemble = ensemble
        self._hidden = hidden

    def forward(self, x):
        """
        Returns ensemble of predictions in shape [Batch x Ensemble].
        """
        # B x E >> B x N*H >> B x N x H >> N x B x H
        x = self.linear1(x).view(-1, self._ensemble, self._hidden).transpose(0, 1)
        x = self.dropout(self.activation(self.layer_norm(x)))
        # N x H >> N x H x 1
        w = self.linear2.weight.unsqueeze(2)
        # N x B x 1 >> N x B >> B x N
        return bmm(x, w).squeeze(-1).transpose(0, 1).contiguous() + self.linear2.bias

    def loss(self, x, y):
        """
        Apply loss function to ensemble of predictions.

        Note: y should be a vector.
        """
        p = self.forward(x) # B x N
        t = y.unsqueeze(-1).expand(p.size())  # B >> B x 1 >> B x N
        return self.loss_function(p, t)

    @no_grad()
    def predict(self, x):
        """
        Average class prediction
        """
        return (sigmoid(self.forward(x)).mean(1) > .5).long()

    @no_grad()
    def predict_proba(self, x, return_std: bool = False):
        """
        Average probability

        :param return_std: return average probability and ensemble standard deviation.
        """
        p = sigmoid(self.forward(x))
        if return_std:
            return p.mean(1), p.std(1)
        return p.mean(1)


__all__ = ['VotingRegressor', 'VotingClassifier', 'BinaryVotingClassifier']
