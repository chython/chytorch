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
from typing import NamedTuple, NamedTupleMeta

try:
    from torch.utils.data._utils.collate import default_collate_fn_map
except ImportError:  # ad-hoc for pytorch<1.13
    default_collate_fn_map = {}


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


__all__ = ['DataTypeMixin', 'NamedTuple', 'default_collate_fn_map']
