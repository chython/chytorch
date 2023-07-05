# -*- coding: utf-8 -*-
#
#  Copyright 2022x, 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from torch import Tensor
from torch.nn import Module
from typing import Tuple, Union


class Slicer(Module):
    def __init__(self, *slc: Union[int, slice, Tuple[int, ...]]):
        """
        Slice input tensor. For use with Sequential.

        E.g. Slicer(slice(None), 0) equal to Tensor[:, 0]
        """
        super().__init__()
        self.slice = slc if len(slc) > 1 else slc[0]

    def forward(self, x: Tensor):
        return x.__getitem__(self.slice)


__all__ = ['Slicer']
