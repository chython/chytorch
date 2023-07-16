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
from functools import cached_property, partial
from json import loads as json_loads
from pickle import loads
from struct import Struct
from torch import Tensor, tensor, float32, Size
from torch.utils.data import Dataset
from typing import List
from zlib import decompress


class StructUnpack(Dataset):
    def __init__(self, data: List[bytes], format_spec: str, dtype=float32):
        """
        Unpack python.struct packed tensors to 1d-tensors.
        Useful in case of highly compressed data.

        :param data: packed data
        :param format_spec: python.struct format for unpacking data
            (e.g. '>bbl' - 2 one-byte ints and 1 big-endian 4 byte int)
        :param dtype: output tensor dtype
        """
        self.data = data
        self.format_spec = format_spec
        self.dtype = dtype
        self._struct = Struct(format_spec)

    def __getitem__(self, item: int) -> Tensor:
        return tensor(self._struct.unpack(self.data[item]), dtype=self.dtype)

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


class TensorUnpack(Dataset):
    def __init__(self, data: List[bytes], dtype=float32):
        """
        Unpack raw tensor byte buffers to 1d-tensors.

        :param data: packed data
        :param dtype: dtype of buffer
        """
        self.data = data
        self.dtype = dtype

    def __getitem__(self, item: int) -> Tensor:
        from torch import frombuffer  # torch>=1.10

        return frombuffer(self.data[item], dtype=self.dtype)

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


class PickleUnpack(Dataset):
    def __init__(self, data: List[bytes]):
        """
        Unpack python-pickled data.

        :param data: packed data
        """
        self.data = data

    def __getitem__(self, item: int):
        return loads(self.data[item])

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


class JsonUnpack(Dataset):
    def __init__(self, data: List[str]):
        """
        Unpack Json data.

        :param data: json strings
        """
        self.data = data

    def __getitem__(self, item: int):
        return json_loads(self.data[item])

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


class Decompress(Dataset):
    def __init__(self, data: List[bytes], method: str = 'zlib', zdict: bytes = None):
        """
        Decompress zipped data.

        :param data: compressed data
        :param method: zlib or zstd
        :param zdict: zstd decompression dictionary
        """
        assert method in ('zlib', 'zstd')
        self.data = data
        self.method = method
        self.zdict = zdict

    def __getitem__(self, item: int) -> bytes:
        return self.decompress(self.data[item])

    @cached_property
    def decompress(self):
        if self.method == 'zlib':
            return decompress
        # zstd
        from pyzstd import decompress as dc, ZstdDict

        if self.zdict is not None:
            return partial(dc, zstd_dict=ZstdDict(self.zdict))
        return dc

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


class Decode(Dataset):
    def __init__(self, data: List[bytes], encoding: str = 'utf8'):
        """
        Bytes to string decoder dataset

        :param data: byte-coded strings
        :param encoding: string encoding
        """
        self.data = data
        self.encoding = encoding

    def __getitem__(self, item: int) -> str:
        return self.data[item].decode(encoding=self.encoding)

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['TensorUnpack', 'StructUnpack', 'PickleUnpack', 'JsonUnpack', 'Decompress', 'Decode']
