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
from torch.nn import Module
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


__all__ = ['PulingHardtanh']
