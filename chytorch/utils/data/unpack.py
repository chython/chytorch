# -*- coding: utf-8 -*-
#
# Copyright 2023, 2024 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from functools import cached_property, partial
from json import loads as json_loads
from pickle import loads
from struct import Struct
from torch import Tensor, tensor, float32, Size, frombuffer
from torch.utils.data import Dataset
from typing import List, Tuple
from zlib import decompress


class StructUnpack(Dataset):
    def __init__(self, data: List[bytes], format_spec: str, dtype=float32, shape: Tuple[int, ...] = None):
        """
        Unpack python.struct packed tensors to 1d-tensors.
        Useful in case of highly compressed data.

        :param data: packed data
        :param format_spec: python.struct format for unpacking data
            (e.g. '>bbl' - 2 one-byte ints and 1 big-endian 4 byte int)
        :param dtype: output tensor dtype
        :param shape: reshape unpacked 1-D tensor
        """
        self.data = data
        self.format_spec = format_spec
        self.dtype = dtype
        self.shape = shape
        self._struct = Struct(format_spec)

    def __getitem__(self, item: int) -> Tensor:
        x = tensor(self._struct.unpack(self.data[item]), dtype=self.dtype)
        if self.shape is not None:
            return x.reshape(self.shape)
        return x

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


class TensorUnpack(Dataset):
    def __init__(self, data: List[bytes], dtype=float32, shape: Tuple[int, ...] = None):
        """
        Unpack raw tensor byte buffers to 1d-tensors.

        :param data: packed data
        :param dtype: dtype of buffer
        :param shape: reshape unpacked 1-D tensor
        """
        self.data = data
        self.dtype = dtype
        self.shape = shape

    def __getitem__(self, item: int) -> Tensor:
        x = frombuffer(self.data[item], dtype=self.dtype)
        if self.shape is not None:
            return x.reshape(self.shape)
        return x

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
