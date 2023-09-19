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
from chython import MoleculeContainer, ReactionContainer, smiles
from torch import Size
from torch.utils.data import Dataset
from typing import Dict, Union, Sequence, Optional, Type


class SMILESDataset(Dataset):
    def __init__(self, data: Sequence[str], *, canonicalize: bool = False, cache: Optional[Dict[int, bytes]] = None,
                 dtype: Union[Type[MoleculeContainer], Type[ReactionContainer]] = MoleculeContainer,
                 unpack: bool = True, ignore_stereo: bool = True, ignore_bad_isotopes: bool = False,
                 keep_implicit: bool = False, ignore_carbon_radicals: bool = False):
        """
        Smiles to chython containers on-the-fly parser dataset.
        Note: SMILES strings or coded structures can be invalid and lead to exception raising.
        Make sure you have validated input.

        :param data: smiles dataset
        :param canonicalize: do standardization (slow, better to prepare data in advance and keep in kekule form)
        :param cache: dict-like object for caching processed data. caching disabled by default.
        :param dtype: expected type of smiles (reaction or molecule)
        :param unpack: return unpacked structure or chython pack
        :param ignore_stereo: Ignore stereo data.
        :param keep_implicit: keep given in smiles implicit hydrogen count, otherwise ignore on valence error.
        :param ignore_bad_isotopes: reset invalid isotope mark to non-isotopic.
        :param ignore_carbon_radicals: fill carbon radicals with hydrogen (X[C](X)X case).
        """
        self.data = data
        self.canonicalize = canonicalize
        self.cache = cache
        self.dtype = dtype
        self.unpack = unpack
        self.ignore_stereo = ignore_stereo
        self.ignore_bad_isotopes = ignore_bad_isotopes
        self.keep_implicit = keep_implicit
        self.ignore_carbon_radicals = ignore_carbon_radicals

    def __getitem__(self, item: int) -> Union[MoleculeContainer, ReactionContainer, bytes]:
        if self.cache is not None and item in self.cache:
            s = self.cache[item]
            if self.unpack:
                return self.dtype.unpack(s)
            return s

        s = smiles(self.data[item], ignore_stereo=self.ignore_stereo, ignore_bad_isotopes=self.ignore_bad_isotopes,
                   keep_implicit=self.keep_implicit, ignore_carbon_radicals=self.ignore_carbon_radicals)
        if not isinstance(s, self.dtype):
            raise TypeError(f'invalid SMILES: {self.dtype} expected, but {type(s)} given')
        if self.canonicalize:
            s.canonicalize()

        if self.cache is not None:
            p = s.pack()
            self.cache[item] = p
            if self.unpack:
                return s
            return p
        if self.unpack:
            return s
        return s.pack()

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['SMILESDataset']
