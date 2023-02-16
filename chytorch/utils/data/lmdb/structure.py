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
from chython import MoleculeContainer, ReactionContainer
from pathlib import Path
from typing import Union, Type
from .mapper import LMDBMapper


class LMDBStructure(LMDBMapper):
    __slots__ = ('dtype',)

    def __init__(self, db: str, *,
                 dtype: Union[Type[MoleculeContainer], Type[ReactionContainer]] = MoleculeContainer,
                 cache: Union[Path, str, None] = None):
        """
        Map LMDB key-value storage to the integer-key - chython structure.
        Note: internally uses python dicts for int to bytes-key mapping and can be huge on big datasets.

        :param db: lmdb dir path
        :param dtype: type of structure
        :param cache: path to cache file for [re]storing index. caching disabled by default.
        """
        super().__init__(db, cache=cache)
        self.dtype = dtype

    def __getitem__(self, item: int) -> Union[MoleculeContainer, ReactionContainer]:
        return self.dtype.unpack(super().__getitem__(item))

    def __getstate__(self):
        state = super().__getstate__()
        state['dtype'] = self.dtype
        return state


__all__ = ['LMDBStructure']
