# -*- coding: utf-8 -*-
#
#  Copyright 2023 Ramil Nugmanov <rnugmano@its.jnj.com>
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
from typing import Union, List


class PostgresMapper(Dataset):
    """
    Map Postgres table/view to the Sequence Dataset of bytestrings.
    """
    def __init__(self, table: str, column: str, index: str, dsn: str, *, cache: Union[Path, str, None] = None):
        """
        `SELECT {index}, {column} FROM {table} WHERE {index} = x`
        or in optimized batch mode
        `SELECT {index}, {column} FROM {table} WHERE {index} in [x, ...]`

        Note: mapper internally uses python list for index to bytes-key mapping and can be huge on big datasets.

        :param table: table name
        :param column: data column name
        :param index: index column name
        :param dsn: credentials in DNS format
        :param cache: path to cache file for [re]storing index. caching disabled by default.
        """
        self.dsn = dsn
        self.table = table
        self.column = column
        self.index = index
        self.cache = cache

        if cache is None:
            return
        if isinstance(cache, str):
            cache = Path(cache)
        if not cache.exists():
            return
        # load existing cache
        with cache.open('rb') as f:
            self.__mapping = load(f)

    def __getitem__(self, item: int):
        with self._connection.cursor() as cur:
            cur.execute(f'SELECT {self.column} FROM {self.table} WHERE {self.index}={self._mapping[item]} LIMIT 1;')
            return bytes(cur.fetchone()[0])

    def __getitems__(self, items: List[int]):
        mapping = self._mapping
        idx = [mapping[x] for x in items]
        with self._connection.cursor() as cur:
            q = ','.join(str(x) for x in idx)
            cur.execute(f"SELECT {self.index}, {self.column} FROM {self.table} WHERE {self.index} IN ({q});")
            data = dict(cur.fetchall())
            return [bytes(data[x]) for x in idx]

    def __len__(self):
        try:
            return len(self._mapping)
        except AttributeError:
            # temporary open db
            with self._connection.cursor() as cur:
                cur.execute(f'SELECT count({self.index}) FROM {self.table} WHERE {self.column} IS NOT NULL;')
                return cur.fetchone()

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError

    @property
    def _mapping(self):
        try:
            return self.__mapping
        except AttributeError:
            # build mapping
            with self._connection.cursor() as cur:
                cur.execute(f'SELECT {self.index} FROM {self.table} WHERE {self.column} IS NOT NULL ORDER BY {self.index};')  # noqa
                self.__mapping = mapping = [x for x, in cur.fetchall()]
            if (cache := self.cache) is not None:  # save to cache
                if isinstance(cache, str):
                    cache = Path(cache)
                with cache.open('wb') as f:
                    dump(mapping, f)
            return mapping

    @property
    def _connection(self):
        try:
            return self.__connection
        except AttributeError:
            from psycopg2 import connect

            self.__connection = db = connect(self.dsn)
            return db

    def __del__(self):
        try:
            self.__connection.close()
        except AttributeError:
            pass
        else:
            del self.__connection

    def __getstate__(self):
        return {'table': self.table, 'column': self.column, 'index': self.index, 'dsn': self.dsn, 'cache': self.cache}

    def __setstate__(self, state):
        self.__init__(**state)


__all__ = ['PostgresMapper']
