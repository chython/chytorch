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
from mmap import mmap, ACCESS_READ, MADV_RANDOM, MADV_SEQUENTIAL
from pathlib import Path
from pickle import load, dump
from re import compile
from torch import Size
from torch.utils.data import Dataset
from typing import Union


class StringMemoryMapper(Dataset):
    """
    Map file with separated lines to list of lines.
    Useful for large datasets of e.g. smiles strings
    """
    def __init__(self, path: Union[Path, str], *, cache: Union[Path, str, None] = None):
        """
        Note: mapper internally uses python list for index lines offsets and can be huge on big datasets.

        :param path: filepath
        :param cache: path to cache file for [re]storing index. caching disabled by default.
        """
        self.path = path
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

    def __getitem__(self, item: int) -> str:
        try:
            data = self._data
        except AttributeError:
            path = self.path
            if isinstance(path, str):
                path = Path(path)
            self._file = file = path.open('rb')
            self._data = data = mmap(file.fileno(), 0, access=ACCESS_READ)
            data.madvise(MADV_RANDOM)  # disable readahead

        try:
            mapping = self._mapping
        except AttributeError:
            # build mapping
            data.madvise(MADV_SEQUENTIAL)  # faster indexation
            self._mapping = mapping = [0]
            mapping.extend(x.span()[1] for x in compile(b'\n').finditer(data))
            data.madvise(MADV_RANDOM)  # disable readahead
            if (cache := self.cache) is not None:  # save to cache
                if isinstance(cache, str):
                    cache = Path(cache)
                with cache.open('wb') as f:
                    dump(mapping, f)

        data.seek(mapping[item])
        return data.readline().strip().decode()

    def __len__(self):
        return len(self._mapping)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError

    def __del__(self):
        try:
            self._data.close()
            self._file.close()
        except AttributeError:
            pass

    def __getstate__(self):
        return {'path': self.path, 'cache': self.cache}

    def __setstate__(self, state):
        self.__init__(**state)


__all__ = ['StringMemoryMapper']
