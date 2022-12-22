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
from torch import cat, stack, Tensor
from torch.nn import Module, ModuleList
from torchtyping import TensorType
from typing import Type, List, Any, Dict, Tuple, Union


class Exponent(Module):
    def __init__(self, base: int = 10, a: float = -1., b: float = 6., c: float = 1.):
        """
        c * base ^ (a * x + b)

        Examples:
             pIC50 > uM: base = 10, a = -1, b = 6, c = 1
             lg ClintRate > T1/2 Clint: base = 10, a = -1, b = 0, c = 1386.3
        """
        super().__init__()
        self.base = base
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x: Tensor) -> Tensor:
        return self.c * self.base ** (self.a * x + self.b)

    def __repr__(self):
        return f'{self.__class__.__name__}(base={self.base}, a={self.a}, b={self.b}, c={self.c})'


class Linear(Module):
    def __init__(self, a: float = 1., b: float = 0.):
        """
        a * x + b
        """
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: Tensor) -> Tensor:
        return self.a * x + self.b

    def __repr__(self):
        return f'{self.__class__.__name__}(a={self.a}, b={self.b})'


class Converters(Module):
    def __init__(self, converters: List[Tuple[Type[Module], Dict[str, Any]]]):
        """
        Convert each element of tensor to a vector of converted values.
        """
        super().__init__()
        self.converters = ModuleList(m(**a) for m, a in converters)

    def forward(self, x: Union[TensorType['batch'], TensorType['batch', 1]]) -> TensorType['batch', 'n_converters']:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        assert x.dim() == 2 and x.size(1) == 1, 'vector or column-vector expected'
        return cat([c(x) for c in self.converters], dim=-1)


class MultiColumnConverters(Module):
    def __init__(self, converters: List[List[Tuple[Type[Module], Dict[str, Any]]]]):
        """
        Convert elements columnwise of matrix to vectors of converted values.
        """
        assert all(len(converters[0]) == len(x) for x in converters), 'only equal converters count supported'
        super().__init__()
        self.converters = ModuleList(Converters(x) for x in converters)

    def forward(self, x: TensorType['batch', 'value']) -> TensorType['batch', 'value', 'n_converters']:
        assert x.dim() == 2, 'matrix expected'
        assert x.size(1) == len(self.converters), 'column count mismatch'
        return stack([c(x) for x, c in zip(x.split(1, dim=1), self.converters)], dim=1)


__all__ = ['Converters', 'MultiColumnConverters', 'Exponent', 'Linear']
