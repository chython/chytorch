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
from chython.periodictable import Element
from random import random, choice
from torch import LongTensor, cat, Size
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map
from torchtyping import TensorType
from typing import Sequence, Union
from .decoder import ReactionDecoderDataset, collate_decoded_reactions
from .._types import DataTypeMixin, NamedTuple


# isometric atoms
# sorted neighbors bonds, atom symbol > atom symbol, hydrogens count

isosteres = {
    (): [('B', 3), ('C', 4), ('N', 3), ('O', 2), ('F', 1),
                   ('Si', 4), ('P', 3), ('S', 2), ('Cl', 1),
                              ('As', 3), ('Se', 2), ('Br', 1),
                                         ('Te', 2), ('I', 1),
         # [BH4-] [NH4+] [OH-] [X-]
         ('B', 4), ('N', 4), ('O', 1), ('F', 0), ('Cl', 0), ('Br', 0), ('I', 0)],

    (1,): [('B', 2), ('C', 3), ('N', 2), ('O', 1), ('F', 0),
                     ('Si', 3), ('P', 2), ('S', 1), ('Cl', 0),
                                ('As', 2), ('Se', 1), ('Br', 0),
                                           ('Te', 1), ('I', 0),
           ('B', 3), ('N', 3), ('N', 1), ('O', 0), ('S', 0)],  # R[BH3-] R[NH3+] R[NH-] R[OS-]

    (2,): [('B', 1), ('C', 2), ('N', 1), ('O', 0), ('P', 1), ('S', 0), ('As', 1), ('Se', 0),
           ('N', 2), ('N', 0)],  # R=[NH2+], =[N-]
    (1, 1): [('B', 1), ('C', 2), ('N', 1), ('O', 0), ('Si', 2), ('P', 1), ('S', 0), ('As', 1), ('Se', 0), ('Te', 0),
             ('B', 2), ('N', 2), ('N', 0)],  # R2[BH2-] R2[NH2+] R2[N-]

    (3,): [('C', 1), ('N', 0), ('P', 0),
           ('C', 0), ('O', 0)],  # [C-]#[O+]
    (1, 2): [('B', 0), ('C', 1), ('N', 0), ('P', 0), ('As', 0), ('Cl', 0), ('Br', 0), ('I', 0),
             ('N', 1), ('O', 0), ('S', 0)],  # =[NH+]- =[O+]- =[S+]-
    (1, 1, 1): [('B', 0), ('C', 1), ('N', 0), ('Si', 1), ('P', 0), ('As', 0), ('Cl', 0), ('Br', 0), ('I', 0),
                ('B', 1), ('N', 1), ('S', 0)],  # R3[BH-] R3[NH+] R3[S+]

    (1, 3): [('C', 0), ('N', 0)],  # #[N+]-
    (2, 2): [('C', 0), ('Si', 0), ('S', 0), ('Se', 0), ('Te', 0), ('N', 0)],  # =[N+]=
    (1, 1, 2): [('C', 0), ('Si', 0), ('S', 0), ('Se', 0), ('Te', 0), ('B', 0), ('N', 0)],  # R2[B+]= =[N+]R2
    (1, 1, 1, 1): [('C', 0), ('Si', 0), ('S', 0), ('Se', 0), ('Te', 0), ('B', 0), ('N', 0), ('P', 0)],  # R4[B-] R4[NP+]

    (1, 2, 2): [('P', 0), ('As', 0), ('Cl', 0), ('Br', 0), ('I', 0)],
    (1, 1, 1, 2): [('P', 0), ('As', 0), ('Cl', 0), ('Br', 0), ('I', 0)],
    (1, 1, 1, 1, 1): [('P', 0), ('As', 0), ('Cl', 0), ('Br', 0), ('I', 0)],

    (2, 2, 2): [('S', 0), ('Se', 0), ('Te', 0)],
    (1, 1, 2, 2): [('S', 0), ('Se', 0), ('Te', 0)],
    (1, 1, 1, 1, 1, 1): [('S', 0), ('Se', 0), ('Te', 0), ('P', 0)],  # [PF6-]

    (1, 2, 2, 2): [('Cl', 0), ('Br', 0), ('I', 0)],
    (1, 1, 1, 2, 2): [('Cl', 0), ('Br', 0), ('I', 0)],
    (1, 1, 1, 1, 1, 2): [('Cl', 0), ('Br', 0), ('I', 0)],
    (1, 1, 1, 1, 1, 1, 1): [('Cl', 0), ('Br', 0), ('I', 0)]
}
isosteres = {(*bs, rp): [x for x in rps if x[0] != rp] for bs, rps in isosteres.items() for rp, _ in rps}


class PermutedReactionDataPoint(NamedTuple):
    reactants_atoms: TensorType['atoms', int]
    reactants_neighbors: TensorType['atoms', int]
    reactants_distances:  TensorType['atoms', 'atoms', int]
    products_atoms: TensorType['atoms', int]
    products_neighbors: TensorType['atoms', int]
    products_distances: TensorType['atoms', 'atoms', int]
    reactants_mask: TensorType['atoms', float]
    products_mask: TensorType['atoms', float]
    replacement: TensorType['atoms', int]


class PermutedReactionDataBatch(NamedTuple, DataTypeMixin):
    atoms: TensorType['batch*2', 'atoms', int]
    neighbors: TensorType['batch*2', 'atoms', int]
    distances: TensorType['batch*2', 'atoms', 'atoms', int]
    reactants_mask: TensorType['batch', 'atoms', float]
    products_mask: TensorType['batch', 'atoms', float]
    replacement: TensorType['batch*atoms', int]


def collate_permuted_reactions(batch, *, collate_fn_map=None) -> PermutedReactionDataBatch:
    """
    Prepares batches of permuted reactions.

    :return: atoms, neighbors, distances, masks, and atoms replacement legend.

    Note: padding not included into legend.
    """
    return PermutedReactionDataBatch(*collate_decoded_reactions([x[:-1] for x in batch]), cat([x[-1] for x in batch]))


default_collate_fn_map[PermutedReactionDataPoint] = collate_permuted_reactions  # add auto_collation to the DataLoader


class PermutedReactionDataset(Dataset):
    def __init__(self, reactions: Sequence[Union[ReactionContainer, bytes]], *, rate: float = .15,
                 max_distance: int = 10, add_cls: bool = True, add_molecule_cls: bool = True,
                 symmetric_cls: bool = True, disable_components_interaction: bool = True,
                 hide_molecule_cls: bool = False, unpack: bool = False):
        """
        Prepare reactions with randomly permuted "organic" atoms in products.

        :param rate: probability of replacement

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

    def __getitem__(self, item: int) -> PermutedReactionDataPoint:
        r = ReactionContainer.unpack(self.reactions[item]) if self.unpack else self.reactions[item].copy()

        labels = [2] if self.add_cls else []
        for m in r.products:
            bonds = m._bonds  # noqa
            hgs = m._hydrogens  # noqa
            if self.add_molecule_cls:
                labels.append(1)
            for n, a in m.atoms():
                labels.append(a.atomic_number + 2)  # True atom
                k = sorted(x.order for x in bonds[n].values())
                k.append(a.atomic_symbol)
                if (p := isosteres.get(tuple(k))) and random() < self.rate:
                    # fake atom
                    s, h = choice(p)
                    a.__class__ = Element.from_symbol(s)
                    hgs[n] = h
        return PermutedReactionDataPoint(
            *ReactionDecoderDataset((r,), max_distance=self.max_distance, add_cls=self.add_cls,
                                    add_molecule_cls=self.add_molecule_cls, symmetric_cls=self.symmetric_cls,
                                    disable_components_interaction=self.disable_components_interaction,
                                    hide_molecule_cls=self.hide_molecule_cls)[0], LongTensor(labels))

    def __len__(self):
        return len(self.reactions)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['PermutedReactionDataset', 'PermutedReactionDataPoint', 'PermutedReactionDataBatch',
           'collate_permuted_reactions']
