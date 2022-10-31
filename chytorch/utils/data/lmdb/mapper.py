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
from pickle import load, dump
from torch import Size
from torch.utils.data import Dataset
from typing import Union


class LMDBMapper(Dataset):
    __slots__ = ('db', 'cache', '_tr', '_mapping')

    def __init__(self, db: 'lmdb.Environment', *, cache: Union[Path, str, None] = None, validate_cache: bool = True):
        """
        Map LMDB key-value storage to the integer-key - value torch Dataset.
        Note: internally uses python dicts for int to bytes-key mapping and can be huge on big datasets.

        :param db: lmdb environment object
        :param cache: path to cache file for [re]storing index. caching disabled by default.
        :param validate_cache: check cache-dataset size mismatch
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
            mapping = load(f)
        assert isinstance(mapping, dict), 'Mapper cache invalid'
        assert not validate_cache or len(mapping) == db.stat()['entries'], 'Mapper cache size mismatch'
        self._mapping = mapping

    def __len__(self):
        try:
            return len(self._mapping)
        except AttributeError:
            return self.db.stat()['entries']

    def __getitem__(self, item: int):
        try:
            tr = self._tr
        except AttributeError:
            self._tr = tr = self.db.begin()

        try:
            mapping = self._mapping
        except AttributeError:
            with tr.cursor() as c:
                # build mapping
                self._mapping = mapping = dict(enumerate(c.iternext(keys=True, values=False)))
            if (cache := self.cache) is not None:  # save to cache
                if isinstance(cache, str):
                    cache = Path(cache)
                with cache.open('wb') as f:
                    dump(mapping, f)

        return tr.get(mapping[item])

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError

    def __del__(self):
        try:
            self._tr.commit()
        except AttributeError:
            pass


__all__ = ['LMDBMapper']
