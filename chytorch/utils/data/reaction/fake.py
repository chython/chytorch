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
from torch import LongTensor, Size
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Union
from .decoder import ReactionDecoderDataset, collate_decoded_reactions
from .._utils import DataTypeMixin, NamedTuple, default_collate_fn_map


class FakedReactionDataPoint(NamedTuple):
    reactants_atoms: TensorType['atoms', int]
    reactants_neighbors: TensorType['atoms', int]
    reactants_distances: TensorType['atoms', 'atoms', int]
    products_atoms: TensorType['atoms', int]
    products_neighbors: TensorType['atoms', int]
    products_distances: TensorType['atoms', 'atoms', int]
    reactants_mask: TensorType['atoms', float]
    products_mask: TensorType['atoms', float]
    fake: int


class FakedReactionDataBatch(NamedTuple, DataTypeMixin):
    atoms: TensorType['batch*2', 'atoms', int]
    neighbors: TensorType['batch*2', 'atoms', int]
    distances: TensorType['batch*2', 'atoms', 'atoms', int]
    reactants_mask: TensorType['batch', 'atoms', float]
    products_mask: TensorType['batch', 'atoms', float]
    fake: TensorType['batch', int]


def collate_faked_reactions(batch, *, collate_fn_map=None) -> FakedReactionDataBatch:
    """
    Prepares batches of faked reactions.

    :return: atoms, neighbors, distances, masks, and fake label.
    """
    return FakedReactionDataBatch(*collate_decoded_reactions([x[:-1] for x in batch]),
                                  LongTensor([x[-1] for x in batch]))


default_collate_fn_map[FakedReactionDataPoint] = collate_faked_reactions  # add auto_collation to the DataLoader


class FakeReactionDataset(Dataset):
    def __init__(self, reactions: Sequence[Union[ReactionContainer, bytes]], *, rate: float = .5,
                 max_distance: int = 10, add_cls: bool = True, add_molecule_cls: bool = True,
                 symmetric_cls: bool = True, disable_components_interaction: bool = True,
                 hide_molecule_cls: bool = False, unpack: bool = False):
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

    def __getitem__(self, item: int) -> FakedReactionDataPoint:
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
        return FakedReactionDataPoint(
            *ReactionDecoderDataset((r,), max_distance=self.max_distance, add_cls=self.add_cls,
                                    add_molecule_cls=self.add_molecule_cls, symmetric_cls=self.symmetric_cls,
                                    disable_components_interaction=self.disable_components_interaction,
                                    hide_molecule_cls=self.hide_molecule_cls)[0], label)

    def __len__(self):
        return len(self.reactions)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['FakeReactionDataset', 'FakedReactionDataPoint', 'FakedReactionDataBatch', 'collate_faked_reactions']
