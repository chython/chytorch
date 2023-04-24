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
from chython import MoleculeContainer
from math import floor
from torch import Size
from torch.utils.data import Dataset
from typing import Sequence
from zlib import decompress


class MoleculeProductDataset(Dataset):
    """
    Lazy enumeration dataset for combinatorial libraries.

    Building blocks should be coded as properly prepared chython packs.
    """
    def __init__(self, *fragments: Sequence[bytes], compressed: bool = True):
        self.fragments = fragments
        self.compressed = compressed

        # calculate lazy product metadata
        self._divs = divs = []
        self._mods = mods = []

        factor = 1
        for x in reversed(fragments):
            s = len(x)
            divs.insert(0, factor)
            mods.insert(0, s)
            factor *= s
        self._size = factor

    def __getitem__(self, item: int) -> MoleculeContainer:
        if item < 0:
            item += self._size
        if item < 0 or item >= self._size:
            raise IndexError

        fragments = [f[floor(item / d) % m] for f, d, m in zip(self.fragments, self._divs, self._mods)]
        if self.compressed:
            fragments = [decompress(x) for x in fragments]
        tmp = [fragments[0]]
        for x in fragments[1:]:
            tmp.append(x[4:])  # skip header and first atom
        return MoleculeContainer.unpack(b''.join(tmp), compressed=False)

    def __len__(self):
        return self._size

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['MoleculeProductDataset']
