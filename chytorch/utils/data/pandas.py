# -*- coding: utf-8 -*-
#
#  Copyright 2022, 2023 Ramil Nugmanov <rnugmano@its.jnj.com>
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
from chython import MoleculeContainer, ReactionContainer
from torch import Size, tensor, Tensor, float32
from torch.utils.data import Dataset
from typing import List, Union, Type


class PandasStructureDataset(Dataset):
    def __init__(self, data, structure: str, *, unpack: bool = False,
                 dtype: Union[Type[MoleculeContainer], Type[ReactionContainer]] = MoleculeContainer):
        """
        Simple wrapper of pandas DataFrame for convenient data accessing.

        :param data: `pandas.DataFrame` with chython Molecule or Reaction objects or packed structures
        :param structure: column name with structures
        :param dtype: type of structure
        :param unpack: unpack molecules or reactions from bytes
        """
        self.data = data[structure].to_numpy()
        self.dtype = dtype
        self.unpack = unpack

    def __getitem__(self, item: int) -> Union[MoleculeContainer, ReactionContainer]:
        x = self.data[item]
        if self.unpack:
            return self.dtype.unpack(x)
        return x

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


class PandasPropertiesDataset(Dataset):
    def __init__(self, data, properties: List[str], *, dtype=float32):
        """
        Simple wrapper of pandas DataFrame for convenient data accessing.

        :param data: `pandas.DataFrame` with chython Molecule or Reaction objects or packed structures
        :param properties: column names with properties
        :param dtype: output tensor dtype
        """
        self.data = data[properties].to_numpy()
        self.dtype = dtype

    def __getitem__(self, item: int) -> Tensor:
        return tensor(self.data[item], dtype=self.dtype)

    def __len__(self):
        return len(self.data)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['PandasStructureDataset', 'PandasPropertiesDataset']
