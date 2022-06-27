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
from chython import ReactionContainer
from itertools import chain, repeat
from torch import IntTensor, cat, zeros, int32, Size
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Tuple, Union
from .molecule import MoleculeDataset


def collate_reactions(batch) -> Tuple[TensorType['batch', 'tokens', int],
                                      TensorType['batch', 'tokens', int],
                                      TensorType['batch', 'tokens', 'tokens', int],
                                      TensorType['batch', 'tokens', int]]:
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
    return pa, pad_sequence(neighbors, True), tmp, pad_sequence(roles, True)


class ReactionDataset(Dataset):
    def __init__(self, reactions: Sequence[Union[ReactionContainer, bytes]], *, distance_cutoff: int = 10,
                 add_cls: bool = True, add_molecule_cls: bool = True, disable_components_interaction: bool = False,
                 unpack: bool = False):
        """
        convert reactions to tuple of:
            atoms, neighbors and distances tensors similar to molecule dataset.
             distances - merged molecular distances matrices filled by zero for isolating attention.
            roles: 2 reactants, 3 products, 0 padding, 1 cls token.

        :param reactions: map-like reactions collection
        :param distance_cutoff: set distances greater than cutoff to cutoff value
        :param add_cls: add special token at first position
        :param disable_components_interaction: treat molecule components as isolated molecules
        :param add_molecule_cls: add special token at first position of each molecule

        Note: symmetric_cls=False parameter unusable due to disabled molecule cls in reaction level.
        """
        self.reactions = reactions
        self.distance_cutoff = distance_cutoff
        self.add_cls = add_cls
        self.add_molecule_cls = add_molecule_cls
        self.disable_components_interaction = disable_components_interaction
        self.unpack = unpack

    def __getitem__(self, item: int) -> Tuple[TensorType['tokens', int], TensorType['tokens', int],
                                              TensorType['tokens', 'tokens', int], TensorType['tokens', int]]:
        rxn = self.reactions[item]
        if self.unpack:
            rxn = ReactionContainer.unpack(rxn)
        molecules = MoleculeDataset(rxn.reactants + rxn.products, distance_cutoff=self.distance_cutoff,
                                    disable_components_interaction=self.disable_components_interaction,
                                    add_cls=self.add_molecule_cls)

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
                roles.append(0)  # disable molecule cls in reaction encoder
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
        return cat(atoms), cat(neighbors), tmp, IntTensor(roles)

    def __len__(self):
        return len(self.reactions)

    def size(self, dim):
        if dim == 0:
            return len(self.reactions)
        elif dim is None:
            return Size((len(self.reactions),))
        raise IndexError


__all__ = ['ReactionDataset', 'collate_reactions']
