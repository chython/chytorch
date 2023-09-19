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
from torch import sigmoid, no_grad
from torch.nn import GELU
from torch.nn.functional import binary_cross_entropy_with_logits
from torchtyping import TensorType
from typing import Union, Optional
from ._kfold import k_fold_mask
from .regressor import VotingRegressor


class BinaryVotingClassifier(VotingRegressor):
    """
    Simple two-layer perceptron with layer normalization and dropout adopted for effective
    ensemble binary classification tasks.
    """
    def __init__(self, ensemble: int = 10, output: int = 1, hidden: int = 256, input: Optional[int] = None,
                 dropout: float = .5, activation=GELU, layer_norm_eps: float = 1e-5,
                 loss_function=binary_cross_entropy_with_logits, norm_first: bool = False):
        super().__init__(ensemble, output, hidden, input, dropout, activation,
                         layer_norm_eps, loss_function, norm_first)

    @no_grad()
    def predict(self, x: TensorType['batch', 'embedding'], *,
                k_fold: Optional[int] = None) -> Union[TensorType['batch', int], TensorType['batch', 'output', int]]:
        """
        Average class prediction

        :param x: features
        :param k_fold: average ensemble according to k-fold trick described in the `loss` method.
        """
        return (self.predict_proba(x, k_fold=k_fold) > .5).long()

    @no_grad()
    def predict_proba(self, x: TensorType['batch', 'embedding'], *,
                      k_fold: Optional[int] = None) -> Union[TensorType['batch', float],
                                                             TensorType['batch', 'output', float]]:
        """
        Average probability

        :param x: features
        :param k_fold: average ensemble according to k-fold trick described in the `loss` method.
        """
        p = sigmoid(self.forward(x))
        if k_fold is not None:
            m = k_fold_mask(k_fold, self._ensemble, x.size(0), True, p.device).bool()  # B x E
            if self._output != 1:
                m.unsqueeze_(1)  # B x 1 x E
            p.masked_fill_(m, nan)
        return p.nanmean(-1)


__all__ = ['BinaryVotingClassifier']
