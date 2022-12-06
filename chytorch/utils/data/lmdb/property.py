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
from pathlib import Path
from struct import Struct
from torch import Tensor, tensor, float32
from typing import Optional, Iterable, Tuple, Union
from .mapper import LMDBMapper


class LMDBProperties(LMDBMapper):
    __slots__ = ('format_spec', 'columns', 'dtype', 'limit', 'map_size', '_struct', '_readonly', '_count')

    def __init__(self, db: str, format_spec: str, *,
                 columns: Optional[Tuple[int, ...]] = None, dtype=float32, limit: int = 1000,
                 map_size=1_000_000_000, cache: Union[Path, str, None] = None, validate_cache: bool = True):
        """
        Map LMDB key-value storage to the integer-key - Tensor value torch Dataset.
        Note: internally uses python dicts for int to bytes-key mapping and can be huge on big datasets.

        :param db: lmdb dir path
        :param cache: path to cache file for [re]storing index. caching disabled by default.
        :param validate_cache: check cache-dataset size mismatch
        :param format_spec: python.Struct format for unpacking data
        :param columns: column indices in data for retrieving
        :param dtype: output tensor dtype
        :param limit: write transaction putting before commit limit
        :param map_size: lmdb map_size
        """
        super().__init__(db, cache=cache, validate_cache=validate_cache)
        self.format_spec = format_spec
        self.columns = columns
        self.dtype = dtype
        self.limit = limit
        self.map_size = map_size
        self._struct = Struct(format_spec)
        self._readonly = True

    def __getitem__(self, item: int) -> Tensor:
        if not self._readonly:
            self._readonly = True
            self.__del__()  # close write transaction/db

        data = super().__getitem__(item)
        p = self._struct.unpack(data)
        if self.columns:
            p = [p[x] for x in self.columns]
        return tensor(p, dtype=self.dtype)

    def __setitem__(self, key: bytes, value: Iterable):
        value = self._struct.pack(*value)
        if self._readonly:  # switch to write mode
            self._readonly = False
            try:
                del self._mapping  # remove mapping if exists
            except AttributeError:
                pass

            self.__del__()  # close and remove transaction

        try:
            tr = self._tr
        except AttributeError:
            from lmdb import Environment

            self._db = db = Environment(self.db, map_size=self.map_size)
            self._tr = tr = db.begin(write=True)
            self._count = 0

        tr.put(key, value)
        # flush transaction
        self._count += 1
        if self._count >= self.limit:
            tr.commit()
            del self._tr


__all__ = ['LMDBProperties']
