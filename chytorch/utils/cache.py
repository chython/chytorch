# -*- coding: utf-8 -*-
#
#  Copyright 2021 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from lmdb import Environment, Transaction
from os import listdir, mkdir
from os.path import join, isdir
from pickle import loads, dumps, load, dump
from typing import Optional


class SequencedFileCache:
    def __init__(self, path):
        if not isdir(path):
            mkdir(path)
            self.key = 0
        else:
            self.key = len(listdir(path))
        self.path = path

    def __iter__(self):
        path = self.path
        for i in range(self.key):
            with open(join(path, str(i)), 'rb') as f:
                yield load(f)

    def __len__(self):
        return self.key

    def append(self, v):
        with open(join(self.path, str(self.key)), 'wb') as f:
            dump(v, f)
        self.key += 1


class SequencedDBCache:
    __slots__ = ('db', 'key', 'limit', 'tr', 'count')

    def __init__(self, db: Environment, *, limit: int = 100):
        self.db = db
        self.key = 0
        self.limit = limit

    def __iter__(self):
        try:
            self.tr.commit()  # close write transaction
        except AttributeError:
            pass
        else:
            del self.tr

        with self.db.begin() as tr, tr.cursor() as c:
            for x in c.iternext(keys=False, values=True):
                yield loads(x)

    def __len__(self):
        return self.key

    def append(self, v):
        k, v = self.key.to_bytes(4, 'big'), dumps(v)
        try:
            tr = self.tr
        except AttributeError:
            self.tr = tr = self.db.begin(write=True)
            self.count = 0

        tr.put(k, v)
        self.key += 1
        self.count += 1
        if self.count >= self.limit:
            tr.commit()
            del self.tr

    def __del__(self):
        try:
            self.tr.commit()
        except AttributeError:
            pass


class SequencedDtypeCompressedCache:
    def __init__(self, compress, decompress, db=None):
        self.db = db if db is not None else []
        self.compress = compress
        self.decompress = decompress

    def __iter__(self):
        decompress = self.decompress
        return (tuple(x.to(t) for x, t in zip(b, decompress)) for b in self.db)

    def __len__(self):
        return len(self.db)

    def append(self, v):
        self.db.append(tuple(x.to(t) for x, t in zip(v, self.compress)))


class CycleDataLoader:
    def __init__(self, loader, cache=None):
        self.loader = loader
        self.cache = cache if cache is not None else []
        self.first_run = True

    def __iter__(self):
        if self.first_run:
            self.iter = iter(self.loader)
            return self
        return iter(self.cache)

    def __next__(self):
        try:
            b = next(self.iter)
        except StopIteration:
            self.first_run = False
            self.iter = None
            raise
        else:
            self.cache.append(b)
            return b

    def __len__(self):
        return len(self.loader)


__all__ = ['SequencedFileCache', 'SequencedDBCache', 'SequencedDtypeCompressedCache',  'CycleDataLoader']
