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
from torch import Size
from torch.utils.data import Dataset


class LMDBMapper(Dataset):
    __slots__ = ('db', '_readonly', '_tr', '_mapping', '_count', 'limit')

    def __init__(self, db: 'lmdb.Environment', *, limit: int = 1000):
        """
        Map LMDB key-value storage to the integer-key - value torch Dataset.
        Note: internally uses python dicts for int to bytes-key mapping and can be huge on big datasets.

        :param db: lmdb environment object
        :param limit: write transaction putting before commit limit
        """
        self.db = db
        self.limit = limit
        self._readonly = True

    def __len__(self):
        try:
            return len(self._mapping)
        except AttributeError:
            return self.db.stat()['entries']

    def __getitem__(self, item: int):
        if not self._readonly:
            self._readonly = True
            try:
                self._tr.commit()  # close write transaction
            except AttributeError:
                pass  # transaction not found
            else:
                del self._tr

        # now we are in readonly mode
        try:
            tr = self._tr
        except AttributeError:
            self._tr = tr = self.db.begin()

        try:
            mp = self._mapping
        except AttributeError:
            with tr.cursor() as c:
                # build mapping
                self._mapping = mp = dict(enumerate(c.iternext(keys=True, values=False)))

        return tr.get(mp[item])

    def __setitem__(self, key: bytes, value: bytes):
        if self._readonly:  # switch to write mode
            self._readonly = False
            try:
                del self._mapping  # remove mapping if exists
            except AttributeError:
                pass

            try:
                self._tr.commit()  # close and remove transaction
            except AttributeError:
                pass
            else:
                del self._tr

        try:
            tr = self._tr
        except AttributeError:
            self._tr = tr = self.db.begin(write=True)
            self._count = 0

        tr.put(key, value)
        # flush transaction
        self._count += 1
        if self._count >= self.limit:
            tr.commit()
            del self._tr

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
