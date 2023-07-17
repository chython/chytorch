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
from pickle import load, dump
from torch import Size
from torch.utils.data import Dataset
from typing import Union


class LMDBMapper(Dataset):
    """
    Map LMDB key-value storage to the Sequence Dataset of bytestrings.
    """
    def __init__(self, db: str, *, cache: Union[Path, str, None] = None):
        """
        Note: mapper internally uses python list for index to bytes-key mapping and can be huge on big datasets.

        :param db: lmdb dir path
        :param cache: path to cache file for [re]storing index. caching disabled by default.
        """
        self.db = db
        self.cache = cache

        if cache is None:
            return
        if isinstance(cache, str):
            cache = Path(cache)
        if not cache.exists():
            return
        # load existing cache
        with cache.open('rb') as f:
            self._mapping = load(f)

    def __getitem__(self, item: int) -> bytes:
        try:
            tr = self._tr
        except AttributeError:
            from lmdb import Environment

            self._db = db = Environment(self.db, readonly=True, lock=False)
            self._tr = tr = db.begin()

        try:
            mapping = self._mapping
        except AttributeError:
            with tr.cursor() as c:
                # build mapping
                self._mapping = mapping = list(c.iternext(keys=True, values=False))
            if (cache := self.cache) is not None:  # save to cache
                if isinstance(cache, str):
                    cache = Path(cache)
                with cache.open('wb') as f:
                    dump(mapping, f)

        return tr.get(mapping[item])

    def __len__(self):
        try:
            return len(self._mapping)
        except AttributeError:
            # temporary open db
            from lmdb import Environment

            with Environment(self.db, readonly=True, lock=False) as f:
                return f.stat()['entries']

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError

    def __del__(self):
        try:
            self._tr.commit()
            self._db.close()
        except AttributeError:
            pass
        else:
            del self._tr, self._db

    def __getstate__(self):
        return {'db': self.db, 'cache': self.cache}

    def __setstate__(self, state):
        self.__init__(**state)


__all__ = ['LMDBMapper']
