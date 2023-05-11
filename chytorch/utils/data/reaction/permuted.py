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
from chython import ReactionContainer
from chython.periodictable import Element
from itertools import chain
from random import random, choice
from torch import IntTensor, Size
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Union


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
    (1, 1, 1, 1, 1, 1, 1): [('Cl', 0), ('Br', 0), ('I', 0)],

    (4, 4): [('C', 1), ('N', 0), ('N', 1), ('O', 0), ('S', 0)],
    (4, 4, 1): [('C', 0), ('N', 0)],
    (4, 4, 4): [('C', 0), ('N', 0)]
}
isosteres = {(*bs, rp): [x for x in rps if x[0] != rp] for bs, rps in isosteres.items() for rp, _ in rps}


class PermutedReactionDataset(Dataset):
    def __init__(self, reactions: Sequence[Union[ReactionContainer, bytes]], *,
                 rate: float = .15, unpack: bool = False):
        """
        Prepare reactions with randomly permuted "organic" atoms.

        :param rate: probability of replacement
        :param unpack: unpack reactions
        """
        self.reactions = reactions
        self.rate = rate
        self.unpack = unpack

    def __getitem__(self, item: int) -> ReactionContainer:
        r = ReactionContainer.unpack(self.reactions[item]) if self.unpack else self.reactions[item].copy()
        for m in chain(r.products, r.reactants):
            bonds = m._bonds  # noqa
            hgs = m._hydrogens  # noqa
            for n, a in m.atoms():
                k = sorted(x.order for x in bonds[n].values())
                k.append(a.atomic_symbol)
                if (p := isosteres.get(tuple(k))) and random() < self.rate:
                    # fake atom
                    s, h = choice(p)
                    a.__class__ = Element.from_symbol(s)
                    hgs[n] = h
        return r

    def __len__(self):
        return len(self.reactions)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


class ReactionLabelsDataset(Dataset):
    def __init__(self, reactions: Sequence[Union[ReactionContainer, bytes]], *,
                 add_cls: bool = True, unpack: bool = False):
        """
        Return atoms' tokens of reactions.

        :param add_cls: add special token at first position
        :param unpack: unpack reactions
        """
        self.reactions = reactions
        self.add_cls = add_cls
        self.unpack = unpack

    def __getitem__(self, item: int) -> TensorType['atoms', int]:
        r = ReactionContainer.unpack(self.reactions[item]) if self.unpack else self.reactions[item]
        labels = [1] if self.add_cls else []
        for m in chain(r.products, r.reactants):
            for _, a in m.atoms():
                labels.append(a.atomic_number + 2)
        return IntTensor(labels)

    def __len__(self):
        return len(self.reactions)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['PermutedReactionDataset', 'ReactionLabelsDataset']
