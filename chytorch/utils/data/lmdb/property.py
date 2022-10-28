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
from struct import Struct
from torch import Tensor, tensor, float32
from typing import Optional, Iterable, Tuple
from .mapper import LMDBMapper


class LMDBProperties(LMDBMapper):
    __slots__ = ('format_spec', 'columns', 'dtype', '_struct')

    def __init__(self, db: 'lmdb.Environment', format_spec: str, *,
                 columns: Optional[Tuple[int, ...]] = None, dtype=float32, limit: int = 1000,):
        """
        Map LMDB key-value storage to the integer-key - Tensor value torch Dataset.
        Note: internally uses python dicts for int to bytes-key mapping and can be huge on big datasets.

        :param db: lmdb environment object
        :param format_spec: python.Struct format for unpacking data
        :param columns: column indices in data for retrieving
        :param dtype: output tensor dtype
        :param limit: write transaction putting before commit limit
        """
        super().__init__(db, limit=limit)
        self.format_spec = format_spec
        self.columns = columns
        self.dtype = dtype
        self._struct = Struct(format_spec)

    def __getitem__(self, item: int) -> Tensor:
        data = super().__getitem__(item)
        p = self._struct.unpack(data)
        if self.columns:
            p = [p[x] for x in self.columns]
        return tensor(p, dtype=self.dtype)

    def __setitem__(self, key: bytes, value: Iterable):
        super().__setitem__(key, self._struct.pack(*value))


__all__ = ['LMDBProperties']
