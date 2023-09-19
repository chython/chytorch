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
