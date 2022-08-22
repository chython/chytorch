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
from random import random
from torch import LongTensor, cat
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Tuple, Union
from .reaction import *


# isometric atoms
# sorted neighbors bonds, atom symbol > atom symbol, hydrogens count

iso_geometry = {('C',): ('O', 2),  # methane - strange case
                (1, 'C'): ('O', 1),
                (2, 'C'): ('O', 0),
                (3, 'C'): ('N', 0),
                (1, 1, 'C'): ('O', 0),
                (1, 2, 'C'): ('N', 0),
                (1, 3, 'C'): ('N', 0),  # charged!
                (2, 2, 'C'): ('N', 0),  # charged!
                (1, 1, 1, 'C'): ('N', 0),
                (1, 1, 2, 'C'): ('N', 0),  # charged!
                (1, 1, 1, 1, 'C'): ('N', 0),  # charged!
                (4, 4, 'C'): ('N', 0), (1, 4, 4, 'C'): ('N', 0), (4, 4, 4, 'C'): ('N', 0),  # aromatic
                (1, 4, 4, 'N'): ('C', 0), (4, 4, 4, 'N'): ('C', 0),

                (1, 3, 'N'): ('C', 0),  # -[N+]# > -C#
                (1, 2, 2, 'P'): ('Cl', 0),
                (1, 1, 1, 2, 'P'): ('Cl', 0),
                (1, 1, 2, 2, 'S'): ('Se', 0),
                (1, 2, 2, 2, 'Cl'): ('I', 0)
                }

replace_map = (((), ('C', 4), ('B', 'Si', 'N', 'P', 'As', 'O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I')),
               # -BH2, -[BH3-], -CH3, -NH2, -[NH3+], -[NH-], -[O-], -OH, -F
               ((1,), ('C', 3), ('B', 'Si', 'N', 'P', 'As', 'O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I')),
               # =BH, =[BH2-], =CH2, =NH, =[NH2+], =[N-], =O
               ((2,), ('C', 2), ('B', 'N', 'P', 'O', 'S', 'Se', 'Te')),
               # #[C-], #CH, #N, [C-]#[O+]
               ((3,), ('C', 1), ('N', 'O')),

               # -BH-, -[BH2-]-, -CH2-, -SiH2-, -NH-, -[NH2+]-, -[N-]-, -PH-, -O-
               ((1, 1), ('C', 2), ('B', 'Si', 'N', 'P', 'As', 'O', 'S', 'Se', 'Te')),
               # -B=, -[BH-]=, -CH=, -N=, -[NH+]=, -[S+]=, -I=
               ((1, 2), ('C', 1), ('B', 'Si', 'N', 'P', 'As', 'S', 'Cl', 'Br', 'I')),
               # =C=, =[N+]=, =S=
               ((2, 2), ('C', 0), ('N', 'S', 'Se', 'Te')),
               ((4, 4), ('C', 1), ('N', 'O', 'S', 'Se')),

               # -B(-)-, -[BH-](-)-, -CH(-)-, -N(-)-, -[NH+](-)-, -[S+](-)-, -Cl(-)-
               ((1, 1, 1), ('C', 1), ('B', 'Si', 'N', 'P', 'As', 'S', 'Cl', 'Br', 'I')),
               # -C(-)=, -[N+](-)=, -S(-)=,
               ((1, 1, 2), ('C', 0), ('Si', 'N', 'S', 'Se', 'Te')),
               # -Cl(=)=
               ((1, 2, 2), ('P', 0), ('As', 'Cl', 'Br', 'I')),

               # -C(-)(-)-, -[N+](-)(-)-, -S(-)(-)-
               ((1, 1, 1, 1), ('C', 0), ('Si', 'N', 'S', 'Se', 'Te')),
               ((1, 1, 1, 2), ('P', 0), ('As', 'Cl', 'Br', 'I')),
               ((1, 1, 2, 2), ('S', 0), ('Se', 'Te')),
               ((1, 2, 2, 2), ('Cl', 0), ('Br', 'I')),
               )


for k, r, ass in replace_map:
    for a in ass:
        iso_geometry[(*k, a)] = r

del k, r, ass, a, replace_map


def collate_permuted_reactions(batch) -> Tuple[TensorType['batch', 'atoms', int],
                                               TensorType['batch', 'atoms', int],
                                               TensorType['batch', 'atoms', 'atoms', int],
                                               TensorType['batch', 'atoms', int],
                                               TensorType['batch*atoms', int]]:
    """
    Prepares batches of permuted reactions.

    :return: atoms, neighbors, distances, atoms roles, and atoms replacement legend.

    Note: cls and padding not included into legend.
    """
    return *collate_reactions([x[:4] for x in batch]), cat([x[-1] for x in batch])


class PermutedReactionDataset(Dataset):
    def __init__(self, reactions: Sequence[Union[ReactionContainer, bytes]], *, rate: float = .15,
                 only_product: bool = False, distance_cutoff: int = 10, add_cls: bool = True,
                 add_molecule_cls: bool = True, symmetric_cls: bool = True,
                 disable_components_interaction: bool = False, hide_molecule_cls: bool = True, unpack: bool = False):
        """
        Prepare reactions with permuted atoms.
        Organic atoms with valence <= 4 can be randomly replaced by carbon.
        Carbons with valence 1,2 and methane can be replaced by oxygen, with valence 3,4 by nitrogen.
        5-valent atoms replaced by P, P(V) > Cl
        6-valent by S, S > Se
        7-valent by Cl, Cl > I.

        :param rate: probability of replacement
        :param only_product: replace only product atoms

        See ReactionDataset for other params description.
        """
        self.rate = rate
        self.only_product = only_product
        self.reactions = reactions
        self.distance_cutoff = distance_cutoff
        self.add_cls = add_cls
        self.add_molecule_cls = add_molecule_cls
        self.symmetric_cls = symmetric_cls
        self.disable_components_interaction = disable_components_interaction
        self.hide_molecule_cls = hide_molecule_cls
        self.unpack = unpack

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, item: int) -> Tuple[TensorType['atoms', int], TensorType['atoms', int],
                                              TensorType['atoms', 'atoms', int], TensorType['atoms', int],
                                              TensorType['atoms', int]]:
        r = ReactionContainer.unpack(self.reactions[item]) if self.unpack else self.reactions[item].copy()

        labels = []
        for m in (r.products if self.only_product else r.molecules()):
            bonds = m._bonds
            hgs = m._hydrogens
            for n, a in m.atoms():
                k = sorted(x.order for x in bonds[n].values())
                k.append(a.atomic_symbol)
                if (p := iso_geometry.get(tuple(k))) and random() < self.rate:
                    s, h = p
                    a.__class__ = Element.from_symbol(s)
                    hgs[n] = h
                    labels.append(0)  # Fake atom
                else:
                    labels.append(1)  # True atom
        return *ReactionDataset((r,), distance_cutoff=self.distance_cutoff, add_cls=self.add_cls,
                                add_molecule_cls=self.add_molecule_cls, symmetric_cls=self.symmetric_cls,
                                disable_components_interaction=self.disable_components_interaction,
                                hide_molecule_cls=self.hide_molecule_cls)[0], LongTensor(labels)


__all__ = ['PermutedReactionDataset', 'collate_permuted_reactions']
