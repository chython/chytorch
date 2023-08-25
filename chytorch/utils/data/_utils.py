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
from random import shuffle
from torch import Size
from torch.utils.data import Dataset
from typing import List, Sequence, TypeVar
from .lmdb import LMDBMapper
from .unpack import Decompress


element = TypeVar('element')


class chained_collate:
    """
    Collate batch of tuples with different data structures by different collate functions.

    :param skip_nones: ignore entities with Nones
    """
    def __init__(self, *collate_fns, skip_nones=True):
        self.collate_fns = collate_fns
        self.skip_nones = skip_nones

    def __call__(self, batch):
        sub_batches = [[] for _ in self.collate_fns]
        for x in batch:
            if self.skip_nones and (x is None or None in x):
                continue
            for y, s in zip(x, sub_batches):
                s.append(y)
        return [f(x) for x, f in zip(sub_batches, self.collate_fns)]


def skip_none_collate(collate_fn):
    def w(batch):
        return collate_fn([x for x in batch if x is not None])
    return w


def load_lmdb(path, size=4, order='big'):
    """
    Helper for loading LMDB datasets with continuous integer keys.
    Note: keys of datapoints should be positive indices coded as bytes with size `size` and `order` endianness.

    Example structure of DB with a key size=2:
        0000 (0): first record
        0001 (1): second record
        ...
        ffff (65535): last record

    :param path: path to the database
    :param size: key size in bytes
    :param order: big or little endian
    """
    db = LMDBMapper(path)
    db._mapping = ByteRange(len(db), size=size, order=order)
    return db


def load_lmdb_zstd_dict(path, size=4, order='big', key=b'\xff\xff\xff\xff'):
    """
    Helper for loading LMDB datasets with continuous integer keys compressed by zstd with external dictionary.
    Note: keys of datapoints should be positive indices coded as bytes with size `size` and `order` endianness.
        Database should contain one additional record with key `key` with decompression dictionary.

    Example structure of DB a key size=2:
        ffff (65535): zstd dict bytes
        0000 (0): first record
        0001 (1): second record
        ...
        fffe (65534): last record

    :param path: path to the database
    :param key: LMDB entry with dictionary data
    :param size: key size in bytes
    :param order: big or little endian
    """
    db = LMDBMapper(path)
    db._mapping = ByteRange(len(db) - 1, size=size, order=order)
    db[0]  # connect db
    dc = Decompress(db, 'zstd', db._tr.get(key))
    return dc


class SizedList(List):
    """
    List with tensor-like size method.
    """
    def size(self, dim=None):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


class ByteRange:
    """
    Range returning values as bytes
    """
    def __init__(self, *args, size=4, order='big', **kwargs):
        self.range = range(*args, **kwargs)
        self.size = size
        self.order = order

    def __getitem__(self, item):
        return self.range[item].to_bytes(self.size, self.order)

    def __len__(self):
        return len(self.range)


class ShuffledList(Dataset):
    """
    Returns randomly shuffled sequences
    """
    def __init__(self, data: Sequence[Sequence[element]]):
        self.data = data

    def __getitem__(self, item: int) -> List[element]:
        x = list(self.data[item])
        shuffle(x)
        return x

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


class SuppressException(Dataset):
    """
    Catch exceptions in wrapped dataset and return None instead
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        try:
            return self.dataset[item]
        except Exception:
            pass

    def __len__(self):
        return len(self.dataset)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['SizedList',
           'ShuffledList',
           'SuppressException',
           'ByteRange',
           'chained_collate', 'skip_none_collate',
           'load_lmdb', 'load_lmdb_zstd_dict']
