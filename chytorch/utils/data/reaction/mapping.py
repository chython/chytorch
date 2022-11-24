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
from torch import Size, IntTensor, BoolTensor, zeros, float32
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Tuple, Union
from .decoder import *


def collate_mapped_reactions(batch) -> Tuple[TensorType['batch*2', 'atoms', int],
                                             TensorType['batch*2', 'atoms', int],
                                             TensorType['batch*2', 'atoms', 'atoms', int],
                                             TensorType['batch', 'atoms', float],
                                             TensorType['batch', 'atoms', float],
                                             TensorType['batch', 'atoms', 'atoms', float],
                                             TensorType['batch', 'atoms', 'atoms', float]]:
    """
    Prepares batches of mapped reactions.

    :return: atoms, neighbors, distances, reaction masks, mapping mask, mapping.
    """
    a, n, d, rm, pm = collate_decoded_reactions([x[:-2] for x in batch])
    b, s = a.shape
    mask = zeros(b // 2, s, s, dtype=float32)
    mapping = zeros(b // 2, s, s, dtype=float32)
    for i, (*_, m, t) in enumerate(batch):
        s1, s2 = m.shape
        mask[i, :s1, :s2] = m
        mapping[i, :s1, :s2] = t
    return a, n, d, rm, pm, mask, mapping


class MappedReactionDataset(Dataset):
    def __init__(self, reactions: Sequence[Union[ReactionContainer, bytes]], *,
                 max_distance: int = 10, add_cls: bool = True, add_molecule_cls: bool = True,
                 symmetric_cls: bool = True, disable_components_interaction: bool = True,
                 hide_molecule_cls: bool = False, unpack: bool = False):
        """
        See ReactionDecoderDataset for params description.
        """
        self.reactions = reactions
        self.max_distance = max_distance
        self.add_cls = add_cls
        self.add_molecule_cls = add_molecule_cls
        self.symmetric_cls = symmetric_cls
        self.disable_components_interaction = disable_components_interaction
        self.hide_molecule_cls = hide_molecule_cls
        self.unpack = unpack

    def __getitem__(self, item: int) -> Tuple[TensorType['atoms', int], TensorType['atoms', int],
                                              TensorType['atoms', 'atoms', int],
                                              TensorType['atoms', int], TensorType['atoms', int],
                                              TensorType['atoms', 'atoms', int],
                                              TensorType['atoms', float], TensorType['atoms', float],
                                              TensorType['atoms', 'atoms', float], TensorType['atoms', 'atoms', float]]:
        r = ReactionContainer.unpack(self.reactions[item]) if self.unpack else self.reactions[item]
        rd = ReactionDecoderDataset((r,), max_distance=self.max_distance, add_cls=self.add_cls,
                                    add_molecule_cls=self.add_molecule_cls, symmetric_cls=self.symmetric_cls,
                                    disable_components_interaction=self.disable_components_interaction,
                                    hide_molecule_cls=self.hide_molecule_cls)[0]
        r_atoms = []
        for m in r.reactants:
            if self.add_molecule_cls:
                r_atoms.append(0)
            r_atoms.extend(m)

        p_atoms = [0] if self.add_cls else []
        for m in r.products:
            if self.add_molecule_cls:
                p_atoms.append(0)
            p_atoms.extend(m)

        common = set(r_atoms).intersection(p_atoms)
        r_atoms = [x if x in common else 0 for x in r_atoms]
        p_atoms = [x if x in common else 0 for x in p_atoms]

        mask = (BoolTensor(p_atoms).unsqueeze(1) & BoolTensor(r_atoms).unsqueeze(0)).float()
        attn = (IntTensor(p_atoms).unsqueeze(1) == IntTensor(r_atoms).unsqueeze(0)).float()
        return *rd, mask, attn * mask

    def __len__(self):
        return len(self.reactions)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['MappedReactionDataset', 'collate_mapped_reactions']
