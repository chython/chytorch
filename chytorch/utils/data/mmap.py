# -*- coding: utf-8 -*-
#
# Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from mmap import mmap, ACCESS_READ
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

    def __getitem__(self, item: int) -> bytes:
        try:
            data = self._data
        except AttributeError:
            path = self.path
            if isinstance(path, str):
                path = Path(path)
            self._file = file = path.open('rb')
            self._data = data = mmap(file.fileno(), 0, access=ACCESS_READ)

            try:
                from mmap import MADV_RANDOM
            except ImportError:  # windows
                pass
            else:
                data.madvise(MADV_RANDOM)  # disable readahead

        try:
            mapping = self._mapping
        except AttributeError:  # build mapping
            try:
                from mmap import MADV_SEQUENTIAL
            except ImportError:  # windows
                pass
            else:
                data.madvise(MADV_SEQUENTIAL)  # faster indexation

            self._mapping = mapping = [0]
            # expected what all lines properly ended
            mapping.extend(x.span()[1] for x in compile(b'\n').finditer(data))
            if (cache := self.cache) is not None:  # save to cache
                if isinstance(cache, str):
                    cache = Path(cache)
                with cache.open('wb') as f:
                    dump(mapping, f)

            try:
                from mmap import MADV_RANDOM
            except ImportError:  # windows
                pass
            else:
                data.madvise(MADV_RANDOM)  # disable readahead
        return data[mapping[item]: mapping[item + 1]]

    def __len__(self):
        return len(self._mapping) - 1

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
