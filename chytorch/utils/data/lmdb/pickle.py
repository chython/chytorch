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
from pathlib import Path
from pickle import loads, dumps
from typing import Union
from .mapper import LMDBMapper


class LMDBPickle(LMDBMapper):
    __slots__ = ('limit', 'map_size', '_struct', '_readonly', '_count')

    def __init__(self, db: str, limit: int = 1000, map_size=1_000_000_000, cache: Union[Path, str, None] = None):
        """
        Map LMDB key-value storage to the integer-key - Tensor value torch Dataset.
        Note: internally uses python dicts for int to bytes-key mapping and can be huge on big datasets.

        :param db: lmdb dir path
        :param cache: path to cache file for [re]storing index. caching disabled by default.
        :param limit: write transaction putting before commit limit
        :param map_size: lmdb map_size
        """
        super().__init__(db, cache=cache)
        self.limit = limit
        self.map_size = map_size
        self._readonly = True

    def __getitem__(self, item: int):
        if not self._readonly:
            self._readonly = True
            self.__del__()  # close write transaction/db

        return loads(super().__getitem__(item))

    def __setitem__(self, key: bytes, value):
        value = dumps(value)
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


__all__ = ['LMDBPickle']
