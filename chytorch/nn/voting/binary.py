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
from torch import sigmoid, no_grad
from torch.nn import GELU
from torch.nn.functional import binary_cross_entropy_with_logits
from torchtyping import TensorType
from typing import Union
from .regressor import VotingRegressor


class BinaryVotingClassifier(VotingRegressor):
    """
    Simple two-layer perceptron with layer normalization and dropout adopted for effective
    ensemble binary classification tasks.
    """
    def __init__(self, ensemble: int = 10, output: int = 1, hidden: int = 256, dropout: float = .5,
                 activation=GELU, layer_norm_eps: float = 1e-5, loss_function=binary_cross_entropy_with_logits):
        super().__init__(ensemble, output, hidden, dropout, activation, layer_norm_eps, loss_function)

    @no_grad()
    def predict(self, x: TensorType['batch', 'embedding']) -> Union[TensorType['batch', int],
                                                                    TensorType['batch', 'output', int]]:
        """
        Average class prediction

        :param x: features
        """
        return (self.predict_proba(x) > .5).long()

    @no_grad()
    def predict_proba(self, x: TensorType['batch', 'embedding'], *,
                      return_std: bool = False) -> Union[TensorType['batch', float],
                                                         TensorType['batch', 'output', float]]:
        """
        Average probability

        :param x: features
        :param return_std: return average probability and ensemble standard deviation.
        """
        p = sigmoid(self.forward(x))
        if return_std:
            return p.mean(-1), p.std(-1)
        return p.mean(-1)


__all__ = ['BinaryVotingClassifier']
