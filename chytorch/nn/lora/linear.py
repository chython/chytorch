# -*- coding: utf-8 -*-
#
#  Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from math import sqrt
from torch import empty, addmm, Tensor
from torch.nn import Linear as tLinear, Parameter, init
from torch.nn.functional import dropout


class Linear(tLinear):
    """
    LoRA wrapped Linear layer.
    """
    def __init__(self, in_features: int, out_features: int, *args, lora_r: int = 0, lora_alpha: float = 1.,
                 lora_dropout: float = 0., **kwargs):
        """
        :param lora_r: LoRA factorization dimension
        :param lora_alpha: LoRA scaling factor
        :param lora_dropout: LoRA input dropout

        See torch.nn.Linear for other params
        """
        super().__init__(in_features, out_features, *args, **kwargs)
        self.lora_r = lora_r
        if lora_r:  # enable lora
            self.weight.requires_grad = False  # freeze main weights
            self.lora_a = Parameter(init.kaiming_uniform_(empty(lora_r, in_features), a=sqrt(5)))
            self.lora_b = Parameter(init.zeros_(empty(out_features, lora_r)))
            self.lora_dropout = lora_dropout
            self.lora_alpha = lora_alpha
            self._lora_scaling = lora_alpha / lora_r

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        if self.lora_r:
            if self.training and self.lora_dropout:
                x = dropout(x, self.lora_dropout)
            a = x @ self.lora_a.transpose(0, 1)
            return addmm(out.flatten(end_dim=-2), a.flatten(end_dim=-2), self.lora_b.transpose(0, 1),
                         alpha=self._lora_scaling).view(out.shape)
        return out

    def merge_lora(self):
        """
        Transform LoRA linear to normal
        """
        if not self.lora_r:
            return
        self.weight.data += (self.lora_b @ self.lora_a) * self._lora_scaling
        self.weight.requires_grad = True
        self.lora_r = 0
        del self.lora_a, self.lora_b, self.lora_dropout, self.lora_alpha, self._lora_scaling

    def extra_repr(self) -> str:
        r = super().extra_repr()
        if self.lora_r:
            return  r + f', lora_r={self.lora_r}, lora_alpha={self.lora_alpha}, lora_dropout={self.lora_dropout}'
        return r


__all__ = ['Linear']
