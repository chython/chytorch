# -*- coding: utf-8 -*-
#
# Copyright 2022, 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
