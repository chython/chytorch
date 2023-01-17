# -*- coding: utf-8 -*-
#
#  Copyright 2021-2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from itertools import chain, repeat
from torch import IntTensor, cat, zeros, int32, Size
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Union
from ..molecule import MoleculeDataset
from .._utils import DataTypeMixin, NamedTuple, default_collate_fn_map


class ReactionEncoderDataPoint(NamedTuple):
    atoms: TensorType['atoms', int]
    neighbors: TensorType['atoms', int]
    distances: TensorType['atoms', 'atoms', int]
    roles: TensorType['atoms', int]


class ReactionEncoderDataBatch(NamedTuple, DataTypeMixin):
    atoms: TensorType['batch', 'atoms', int]
    neighbors: TensorType['batch', 'atoms', int]
    distances: TensorType['batch', 'atoms', 'atoms', int]
    roles: TensorType['batch', 'atoms', int]


def collate_encoded_reactions(batch, *, collate_fn_map=None) -> ReactionEncoderDataBatch:
    """
    Prepares batches of reactions.

    :return: atoms, neighbors, distances, atoms roles.
    """
    atoms, neighbors, distances, roles = [], [], [], []
    for a, n, d, r in batch:
        atoms.append(a)
        neighbors.append(n)
        distances.append(d)
        roles.append(r)

    pa = pad_sequence(atoms, True)
    b, s = pa.shape
    tmp = zeros(b, s, s, dtype=int32)
    tmp[:, :, 0] = 1  # prevent nan in MHA softmax on padding
    for n, d in enumerate(distances):
        s = d.size(0)
        tmp[n, :s, :s] = d
    return ReactionEncoderDataBatch(pa, pad_sequence(neighbors, True), tmp, pad_sequence(roles, True))


default_collate_fn_map[ReactionEncoderDataPoint] = collate_encoded_reactions  # add auto_collation to the DataLoader


class ReactionEncoderDataset(Dataset):
    def __init__(self, reactions: Sequence[Union[ReactionContainer, bytes]], *, max_distance: int = 10,
                 max_neighbors: int = 14, add_cls: bool = True, add_molecule_cls: bool = True,
                 symmetric_cls: bool = True, disable_components_interaction: bool = False,
                 hide_molecule_cls: bool = True, unpack: bool = False, distance_cutoff=None):
        """
        convert reactions to tuple of:
            atoms, neighbors and distances tensors similar to molecule dataset.
             distances - merged molecular distances matrices filled by zero for isolating attention.
            roles: 2 reactants, 3 products, 0 padding, 1 cls token.

        :param reactions: reactions collection
        :param max_distance: set distances greater than cutoff to cutoff value
        :param add_cls: add special token at first position
        :param add_molecule_cls: add special token at first position of each molecule
        :param symmetric_cls: do bidirectional attention of molecular cls to atoms and back
        :param disable_components_interaction: treat molecule components as isolated molecules
        :param hide_molecule_cls: disable molecule cls in reaction lvl (mark as padding)
        :param max_neighbors: set neighbors count greater than cutoff to cutoff value
        """
        if not add_molecule_cls:
            assert not hide_molecule_cls, 'add_molecule_cls should be True if hide_molecule_cls is True'
            assert not symmetric_cls, 'add_molecule_cls should be True if symmetric_cls is True'
        self.reactions = reactions
        # distance_cutoff is deprecated
        self.max_distance = distance_cutoff if distance_cutoff is not None else max_distance
        self.add_cls = add_cls
        self.add_molecule_cls = add_molecule_cls
        self.symmetric_cls = symmetric_cls
        self.disable_components_interaction = disable_components_interaction
        self.hide_molecule_cls = hide_molecule_cls
        self.max_neighbors = max_neighbors
        self.unpack = unpack

    def __getitem__(self, item: int) -> ReactionEncoderDataPoint:
        rxn = self.reactions[item]
        if self.unpack:
            rxn = ReactionContainer.unpack(rxn)
        molecules = MoleculeDataset(rxn.reactants + rxn.products, max_distance=self.max_distance,
                                    max_neighbors=self.max_neighbors, add_cls=self.add_molecule_cls,
                                    symmetric_cls=self.symmetric_cls,
                                    disable_components_interaction=self.disable_components_interaction)

        if self.add_cls:
            # disable rxn cls in molecules encoder
            atoms, neighbors, roles = [IntTensor([0])], [IntTensor([0])], [1]
        else:
            atoms, neighbors, roles = [], [], []
        distances = []
        for i, (m, r) in enumerate(chain(zip(rxn.reactants, repeat(2)), zip(rxn.products, repeat(3)))):
            a, n, d = molecules[i]
            atoms.append(a)
            neighbors.append(n)
            distances.append(d)
            if self.add_molecule_cls:
                # (dis|en)able molecule cls in reaction encoder
                roles.append(0 if self.hide_molecule_cls else r)
            roles.extend(repeat(r, len(m)))

        tmp = zeros(len(roles), len(roles), dtype=int32)
        if self.add_cls:
            tmp[0, 0] = 1  # prevent nan in MHA softmax.
            i = 1
        else:
            i = 0
        for d in distances:
            j = i + d.size(0)
            tmp[i:j, i:j] = d
            i = j
        return ReactionEncoderDataPoint(cat(atoms), cat(neighbors), tmp, IntTensor(roles))

    def __len__(self):
        return len(self.reactions)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['ReactionEncoderDataset', 'ReactionEncoderDataPoint', 'ReactionEncoderDataBatch',
           'collate_encoded_reactions']
