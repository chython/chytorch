# -*- coding: utf-8 -*-
#
# Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
