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
from torch import LongTensor, cat
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Tuple, Union
from .decoder import *


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


def collate_permuted_reactions(batch) -> Tuple[TensorType['batch*2', 'atoms', int],
                                               TensorType['batch*2', 'atoms', int],
                                               TensorType['batch*2', 'atoms', 'atoms', int],
                                               TensorType['batch', 'atoms', float],
                                               TensorType['batch', 'atoms', float],
                                               TensorType['batch*atoms', int]]:
    """
    Prepares batches of permuted reactions.

    :return: atoms, neighbors, distances, masks, and atoms replacement legend.

    Note: cls and padding not included into legend.
    """
    return *collate_decoded_reactions([x[:-1] for x in batch]), cat([x[-1] for x in batch])


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

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, item: int) -> Tuple[TensorType['atoms', int], TensorType['atoms', int],
                                              TensorType['atoms', 'atoms', int],
                                              TensorType['atoms', int], TensorType['atoms', int],
                                              TensorType['atoms', 'atoms', int],
                                              TensorType['atoms', float], TensorType['atoms', float],
                                              TensorType['atoms', int]]:
        r = ReactionContainer.unpack(self.reactions[item]) if self.unpack else self.reactions[item].copy()

        labels = [1] if self.add_cls else []
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
        return *ReactionDecoderDataset((r,), max_distance=self.max_distance, add_cls=self.add_cls,
                                       add_molecule_cls=self.add_molecule_cls, symmetric_cls=self.symmetric_cls,
                                       disable_components_interaction=self.disable_components_interaction,
                                       hide_molecule_cls=self.hide_molecule_cls)[0], LongTensor(labels)


__all__ = ['PermutedReactionDataset', 'collate_permuted_reactions']
