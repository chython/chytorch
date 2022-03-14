# -*- coding: utf-8 -*-
#
#  Copyright 2021, 2022 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from .molecule import *
from .reaction import *


def chained_collate(*collate_fns):
    def w(batch):
        sub_batches = [[] for _ in collate_fns]
        for x in batch:
            for y, s in zip(x, sub_batches):
                s.append(y)
        return [f(x) for x, f in zip(sub_batches, collate_fns)]
    return w


__all__ = ['MoleculeDataset', 'ReactionDataset', 'collate_molecules', 'collate_reactions', 'chained_collate']
