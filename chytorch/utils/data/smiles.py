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
from chython import MoleculeContainer, ReactionContainer, smiles
from torch import Size
from torch.utils.data import Dataset
from typing import Dict, Union, Sequence, Optional, Type


class SMILESDataset(Dataset):
    def __init__(self, data: Sequence[str], *, canonicalize: bool = False, cache: Optional[Dict[int, bytes]] = None,
                 dtype: Union[Type[MoleculeContainer], Type[ReactionContainer]] = MoleculeContainer):
        """
        Smiles to chython containers on-the-fly parser dataset.
        Note: SMILES strings or coded structures can be invalid and lead to exception raising.
        Make sure you have validated input.

        :param data: smiles dataset
        :param canonicalize: do standardization (slow, better to prepare data in advance and keep in kekule form)
        :param cache: dict-like object for caching processed data. caching disabled by default.
        :param dtype: expected type of smiles (reaction or molecule)
        """
        self.data = data
        self.canonicalize = canonicalize
        self.cache = cache
        self.dtype = dtype

    def __getitem__(self, item: int) -> Union[MoleculeContainer, ReactionContainer]:
        if self.cache is not None and item in self.cache:
            return self.dtype.unpack(self.cache[item])

        s = smiles(self.data[item])
        if not isinstance(s, self.dtype):
            raise TypeError(f'invalid SMILES: {self.dtype} expected, but {type(s)} given')
        if self.canonicalize:
            s.canonicalize()
        if self.cache is not None:
            self.cache[item] = s.pack()
        return s

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['SMILESDataset']
