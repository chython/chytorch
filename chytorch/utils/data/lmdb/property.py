# -*- coding: utf-8 -*-
#
#  Copyright 2022, 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from pathlib import Path
from pickle import loads
from struct import Struct
from torch import Tensor, tensor, float32, frombuffer
from typing import Optional, Union, List
from .mapper import LMDBMapper


class LMDBStruct(LMDBMapper):
    """
    Map LMDB key-value storage of python.struct packed tensors to the Sequence Dataset of 1d-tensors.
    Useful in case of highly compressed data.
    """
    __slots__ = ('format_spec', 'columns', 'dtype', '_struct')

    def __init__(self, db: str, format_spec: str, *,
                 columns: Optional[List[int]] = None, dtype=float32, cache: Union[Path, str, None] = None):
        """
        Note: internally uses python list for index to bytes-key mapping and can be huge on big datasets.

        :param db: lmdb dir path
        :param cache: path to cache file for [re]storing index. caching disabled by default.
        :param format_spec: python.struct format for unpacking data (e.g. '>bbl' - 2 one-byte ints and 1 big-endian 4 byte int)
        :param columns: column indices in data for retrieving
        :param dtype: output tensor dtype
        """
        super().__init__(db, cache=cache)
        self.format_spec = format_spec
        self.columns = columns
        self.dtype = dtype
        self._struct = Struct(format_spec)

    def __getitem__(self, item: int) -> Tensor:
        p = self._struct.unpack(super().__getitem__(item))
        if self.columns:
            p = [p[x] for x in self.columns]
        return tensor(p, dtype=self.dtype)

    def __getstate__(self):
        state = super().__getstate__()
        state['format_spec'] = self.format_spec
        state['columns'] = self.columns
        state['dtype'] = self.dtype
        return state


class LMDBTensor(LMDBMapper):
    """
    Map LMDB key-value storage of tensors buffers to the Sequence Dataset of 1d-tensors.
    """
    __slots__ = ('packed_dtype', 'columns', 'dtype')

    def __init__(self, db: str, packed_dtype=float32, *,
                 columns: Optional[List[int]] = None, dtype=float32, cache: Union[Path, str, None] = None):
        """
        Note: internally uses python list for index to bytes-key mapping and can be huge on big datasets.

        :param db: lmdb dir path
        :param cache: path to cache file for [re]storing index. caching disabled by default.
        :param packed_dtype: dtype of buffer
        :param columns: column indices in data for retrieving
        :param dtype: output tensor dtype
        """
        super().__init__(db, cache=cache)
        self.packed_dtype = packed_dtype
        self.columns = columns
        self.dtype = dtype

    def __getitem__(self, item: int) -> Tensor:
        p = frombuffer(super().__getitem__(item), dtype=self.packed_dtype).to(self.dtype)
        if self.columns:
            return p[self.columns]
        return p

    def __getstate__(self):
        state = super().__getstate__()
        state['packed_dtype'] = self.packed_dtype
        state['columns'] = self.columns
        state['dtype'] = self.dtype
        return state


class LMDBPickle(LMDBMapper):
    """
    Map LMDB key-value storage of pickled python objects to the Sequence Dataset of objects.
    """
    def __getitem__(self, item: int):
        return loads(super().__getitem__(item))


__all__ = ['LMDBStruct', 'LMDBTensor', 'LMDBPickle']
