# -*- coding: utf-8 -*-
#
# Copyright 2023, 2024 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from torch import float32, zeros_like, exp, Tensor
from torch.nn import Parameter, MSELoss
from torch.nn.modules.loss import _Loss
from torchtyping import TensorType


class MultiTaskLoss(_Loss):
    """
    Auto-scalable loss for multitask training.

    https://arxiv.org/abs/1705.07115
    """
    def __init__(self, loss_type: TensorType['loss_type', bool], *, reduction='mean'):
        """
        :param loss_type: vector equal to the number of tasks losses. True for regression and False for classification.
        """
        super().__init__(reduction=reduction)
        self.log = Parameter(zeros_like(loss_type, dtype=float32))
        self.register_buffer('coefficient', (loss_type + 1.).to(float32))

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


class CensoredLoss(_Loss):
    """
    Loss wrapper masking input under target qualifier range.

    Masking strategy for different qualifiers (rows) and input-target relations (columns) described below:

           | I < T | I = T | I > T |
        ---|-------|-------|-------|
        -1 | Mask  |       |       |
         0 |       |       |       |
         1 |       |       | Mask  |

    Wrapped loss should not be configured with mean reduction.
    Wrapper does proper mean reduction by ignoring masked values.

    Note: wrapped loss should correctly treat zero-valued input and targets.
    """
    def __init__(self, loss: _Loss, reduction: str = 'mean', eps: float = 1e-5):
        assert loss.reduction != 'mean', 'given loss should not be configured to `mean` reduction'
        super().__init__(reduction=reduction)
        self.loss = loss
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, qualifier: Tensor) -> Tensor:
        mask = ((qualifier >= 0) | (input >= target)) & ((qualifier <= 0) | (input <= target))
        loss = self.loss(input * mask, target * mask)
        if self.reduction == 'mean':
            if self.loss.reduction == 'none':
                loss = loss.sum()
            return loss / (mask.sum() + self.eps)
        elif self.reduction == 'sum':
            if self.loss.reduction == 'none':
                return loss.sum()
            return loss
        return loss  # reduction='none'


class MaskedNaNLoss(_Loss):
    """
    Loss wrapper masking nan targets and corresponding input values as zeros.
    Wrapped loss should not be configured with mean reduction.
    Wrapper does proper mean reduction by ignoring masked values.

    Note: wrapped loss should correctly treat zero-valued input and targets.
    """
    def __init__(self, loss: _Loss, reduction: str = 'mean', eps: float = 1e-5):
        assert loss.reduction != 'mean', 'given loss should not be configured to `mean` reduction'
        super().__init__(reduction=reduction)
        self.loss = loss
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mask = ~target.isnan()
        loss = self.loss(input * mask, target.nan_to_num())
        if self.reduction == 'mean':
            if self.loss.reduction == 'none':
                loss = loss.sum()
            return loss / (mask.sum() + self.eps)
        elif self.reduction == 'sum':
            if self.loss.reduction == 'none':
                return loss.sum()
            return loss
        return loss  # reduction='none'


class MSLELoss(MSELoss):
    r"""
    Mean Squared Logarithmic Error:

    .. math:: \text{MSLE} = \frac{1}{N}\sum_i^N (\log_e(1 + y_i) - \log_e(1 + \hat{y_i}))^2

    Note: Works only for positive target values range. Implicitly clamps negative input.
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward((input.clamp(min=0) + 1).log(), (target + 1).log())


__all__ = ['MultiTaskLoss',
           'CensoredLoss',
           'MaskedNaNLoss',
           'MSLELoss']
