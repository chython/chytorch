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
from torch import float32, zeros_like, exp
from torch.nn import Module, Parameter
from torchtyping import TensorType
from .functional import puling_hardtanh


class PulingHardtanh(Module):
    """
    Hardtanh with inside-range puling gradient
    """
    def __init__(self, mn, mx):
        super().__init__()
        self.mn = mn
        self.mx = mx

    def forward(self, x):
        return puling_hardtanh(x, self.mn, self.mx)


class MultiTaskLoss(Module):
    """
    Auto-scalable loss for multitask training.

    https://arxiv.org/abs/1705.07115
    """
    def __init__(self, loss_type: TensorType['loss_type', bool], *, reduction='mean'):
        """
        :param loss_type: vector equal to the number of tasks losses. True for regression and False for classification.
        """
        super().__init__()
        self.log = Parameter(zeros_like(loss_type, dtype=float32))
        self.register_buffer('coefficient', (loss_type + 1.).to(float32))
        self.reduction = reduction

    def forward(self, x: TensorType['loss', float]):
        """
        :param x: 1d vector of losses or 2d matrix of batch X losses.
        """
        x = x / (self.coefficient * exp(self.log)) + self.log / 2

        if self.reduction == 'sum':
            return x.sum()
        elif self.reduction == 'mean':
            return x.mean()
        return x


__all__ = ['PulingHardtanh', 'MultiTaskLoss']
