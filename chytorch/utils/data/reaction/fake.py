# -*- coding: utf-8 -*-
#
#  Copyright 2022 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from chython import ReactionContainer
from random import random, choice
from torch import LongTensor
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Tuple, Union
from .decoder import *


def collate_faked_reactions(batch) -> Tuple[TensorType['batch*2', 'atoms', int],
                                            TensorType['batch*2', 'atoms', int],
                                            TensorType['batch*2', 'atoms', 'atoms', int],
                                            TensorType['batch', 'atoms', float],
                                            TensorType['batch', int]]:
    """
    Prepares batches of faked reactions.

    :return: atoms, neighbors, distances, masks, and fake label.
    """
    return *collate_decoded_reactions([x[:-1] for x in batch]), LongTensor([x[-1] for x in batch])


class FakeReactionDataset(Dataset):
    def __init__(self, reactions: Sequence[Union[ReactionContainer, bytes]], *, rate: float = .5,
                 max_distance: int = 10, add_cls: bool = True, add_molecule_cls: bool = True,
                 symmetric_cls: bool = True, disable_components_interaction: bool = False,
                 hide_molecule_cls: bool = True, unpack: bool = False):
        """
        Prepare reactions with switched product and reactant molecules.

        Note: possible to switch only reactions with at least 2 reactants to prevent false-negatives.

        :param rate: probability of switch

        See ReactionDecoderDataset for other params description.
        """
        self.rate = rate
        self.reactions = reactions
        self.max_distance = max_distance
        self.add_cls = add_cls
        self.add_molecule_cls = add_molecule_cls
        self.symmetric_cls = symmetric_cls
        self.disable_components_interaction = disable_components_interaction
        self.hide_molecule_cls = hide_molecule_cls
        self.unpack = unpack

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, item: int) -> Tuple[TensorType['atoms', int], TensorType['atoms', int],
                                              TensorType['atoms', 'atoms', int],
                                              TensorType['atoms', int], TensorType['atoms', int],
                                              TensorType['atoms', 'atoms', int],
                                              TensorType['atoms', float],
                                              int]:
        r = ReactionContainer.unpack(self.reactions[item]) if self.unpack else self.reactions[item].copy()

        if len(r.reactants) > 1 and random() < self.rate:
            rs = list(r.reactants)
            ps = list(r.products)
            ri = choice(range(len(rs)))
            pi = choice(range(len(ps)))
            ps.append(rs.pop(ri))
            rs.append(ps.pop(pi))
            r = ReactionContainer(rs, ps)
            label = 0
        else:
            label = 1
        return *ReactionDecoderDataset((r,), max_distance=self.max_distance, add_cls=self.add_cls,
                                       add_molecule_cls=self.add_molecule_cls, symmetric_cls=self.symmetric_cls,
                                       disable_components_interaction=self.disable_components_interaction,
                                       hide_molecule_cls=self.hide_molecule_cls)[0], label


__all__ = ['FakeReactionDataset', 'collate_faked_reactions']
