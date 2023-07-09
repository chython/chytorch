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
from typing import List, NamedTuple, NamedTupleMeta, Sequence, TypeVar

try:
    from torch.utils.data._utils.collate import default_collate_fn_map
except ImportError:  # ad-hoc for pytorch<1.13
    default_collate_fn_map = {}

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


# https://stackoverflow.com/a/50369521
if hasattr(NamedTuple, '__mro_entries__'):
    # Python 3.9 fixed and broke multiple inheritance in a different way
    # see https://github.com/python/cpython/issues/88089
    from typing import _NamedTuple

    NamedTuple = _NamedTuple


class MultipleInheritanceNamedTupleMeta(NamedTupleMeta):
    def __new__(mcls, typename, bases, ns):
        if NamedTuple in bases:
            base = super().__new__(mcls, '_base_' + typename, bases, ns)
            bases = (base, *(b for b in bases if not isinstance(b, NamedTuple)))
        return super(NamedTupleMeta, mcls).__new__(mcls, typename, bases, ns)


class DataTypeMixin(metaclass=MultipleInheritanceNamedTupleMeta):
    def to(self, *args, **kwargs):
        return type(self)(*(x.to(*args, **kwargs) for x in self))

    def cpu(self, *args, **kwargs):
        return type(self)(*(x.cpu(*args, **kwargs) for x in self))

    def cuda(self, *args, **kwargs):
        return type(self)(*(x.cuda(*args, **kwargs) for x in self))


__all__ = ['SizedList', 'ShuffledList', 'SuppressException', 'ByteRange', 'chained_collate', 'skip_none_collate']
