# -*- coding: utf-8 -*-
#
# Copyright 2022, 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from math import nan
from torch import bmm, no_grad, Tensor
from torch.nn import Dropout, GELU, LayerNorm, LazyLinear, Linear, Module
from torch.nn.functional import smooth_l1_loss
from torchtyping import TensorType
from typing import Optional, Union
from ._kfold import k_fold_mask


class VotingRegressor(Module):
    """
    Simple two-layer perceptron with layer normalization and dropout adopted for effective ensemble regression modeling.
    """
    def __init__(self, ensemble: int = 10, output: int = 1, hidden: int = 256, input: Optional[int] = None,
                 dropout: float = .5, activation=GELU, layer_norm_eps: float = 1e-5, loss_function=smooth_l1_loss,
                 norm_first: bool = False):
        """
        :param ensemble: number of predictive heads per output
        :param input: input features size. By-default do lazy initialization
        :param output: number of predicted properties in multitask mode. By-default single task mode is active
        :param norm_first: do normalization of input
        """
        assert ensemble > 0 and output > 0, 'ensemble and output should be positive integers'
        super().__init__()
        if input is None:
            self.linear1 = LazyLinear(hidden * ensemble * output)
            assert not norm_first, 'input size required for prenormalization'
        else:
            if norm_first:
                self.norm_first = LayerNorm(input, layer_norm_eps)
            self.linear1 = Linear(input, hidden * ensemble * output)

        self.layer_norm = LayerNorm(hidden, layer_norm_eps)
        self.activation = activation()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(hidden, ensemble * output)
        self.loss_function = loss_function

        self._ensemble = ensemble
        self._input = input
        self._hidden = hidden
        self._output = output
        self._norm_first = norm_first

    def forward(self, x):
        """
        Returns ensemble of predictions in shape [Batch x Output*Ensemble].
        """
        if self._norm_first:
            x = self.norm_first(x)
        # B x E >> B x N*H >> B x N x H >> N x B x H
        x = self.linear1(x).view(-1, self._ensemble * self._output, self._hidden).transpose(0, 1)
        x = self.dropout(self.activation(self.layer_norm(x)))
        # N x H >> N x H x 1
        w = self.linear2.weight.unsqueeze(2)
        # N x B x 1 >> N x B >> B x N
        x = bmm(x, w).squeeze(-1).transpose(0, 1).contiguous() + self.linear2.bias
        if self._output != 1:
            return x.view(-1, self._output, self._ensemble)  # B x O x E
        return x  # B x E

    def loss(self, x: TensorType['batch', 'embedding'],
             y: Union[TensorType['batch', 1, float], TensorType['batch', 'output', float]],
             k_fold: Optional[int] = None) -> Tensor:
        """
        Apply loss function to ensemble of predictions.

        Note: y should be a column vector in single task or Batch x Output matrix in multitask mode.

        :param x: features
        :param y: properties
        :param k_fold: Cross-validation training procedure of ensemble model.
            Only k - 1 / k heads of ensemble do train on k - 1 / k items of each batch.
            On validation step 1 / k of same batches used to evaluate heads.
            Batch and ensemble sizes should be divisible by k. Disabled by default.
        """
        p = self.forward(x)
        if self._output != 1:  # MT mode
            y = y.unsqueeze(-1)  # B x O > B x O x 1
        y = y.expand(p.size())  # B x E or B x O x E

        if k_fold is not None:
            m = k_fold_mask(k_fold, self._ensemble, x.size(0), self.training, p.device)  # B x E
            if self._output != 1:
                m = m.unsqueeze(1)  # B x 1 x E
            p = p * m  # zeros in mask disable gradients
            y = y * m  # disable errors in test/val loss
        return self.loss_function(p, y)

    @no_grad()
    def predict(self, x: TensorType['batch', 'embedding'], *,
                k_fold: Optional[int] = None) -> Union[TensorType['batch', float],
                                                       TensorType['batch', 'output', float]]:
        """
        Average prediction

        :param x: features.
        :param k_fold: average ensemble according to k-fold trick described in the `loss` method.
        """
        p = self.forward(x)
        if k_fold is not None:
            m = k_fold_mask(k_fold, self._ensemble, x.size(0), True, p.device).bool()  # B x E
            if self._output != 1:
                m = m.unsqueeze(1)  # B x 1 x E
            p.masked_fill_(m, nan)
        return p.nanmean(-1)


__all__ = ['VotingRegressor']
