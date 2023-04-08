# -*- coding: utf-8 -*-
#
#  Copyright 2022, 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
