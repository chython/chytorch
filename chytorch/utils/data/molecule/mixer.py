# -*- coding: utf-8 -*-
#
#  Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from chython import MoleculeContainer
from torch import IntTensor, cat, zeros, int32, Size, tril, arange, clip, ones
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Union, Hashable, Dict, Optional
from .encoder import collate_molecules
from ..molecule import MoleculeDataset
from .._utils import DataTypeMixin, NamedTuple, default_collate_fn_map


class MoleculeMixerDataPoint(NamedTuple):
    atoms: TensorType['atoms+conditions', int]
    neighbors: TensorType['atoms+conditions', int]
    distances: TensorType['atoms+conditions', 'atoms+conditions', int]
    causal: TensorType['atoms+conditions', int]


class MoleculeMixerDataBatch(NamedTuple, DataTypeMixin):
    atoms: TensorType['batch', 'atoms+conditions', int]
    neighbors: TensorType['batch', 'atoms+conditions', int]
    distances: TensorType['batch', 'atoms+conditions', 'atoms+conditions', int]
    causal: TensorType['batch', 'atoms+conditions', int]


def collate_mixed_molecules(batch, *, padding_left: bool = False, collate_fn_map=None) -> MoleculeMixerDataBatch:
    """
    Prepares batches of molecules.

    :return: atoms, neighbors, distances, roles, causal.
    """
    if padding_left:
        causal = pad_sequence([x[-1].flipud() for x in batch], True).fliplr()
    else:
        causal = pad_sequence([x[-1] for x in batch], True)
    return MoleculeMixerDataBatch(*collate_molecules([x[:-1] for x in batch], padding_left=padding_left), causal)  # noqa


default_collate_fn_map[MoleculeMixerDataPoint] = collate_mixed_molecules  # add auto_collation to the DataLoader


class MoleculeMixerDataset(Dataset):
    def __init__(self, molecules: Sequence[Union[MoleculeContainer, bytes]], conditions: Sequence[Sequence[Hashable]],
                 *, max_distance: int = 10, max_neighbors: int = 14, add_cls: bool = True, unpack: bool = False,
                 dictionary: Dict[Hashable, int] = None, positional_distance: int = 10,
                 max_sequence: Optional[int] = None):
        """
        convert molecules and categorical conditions to tuple of:
            atoms, neighbors and distances tensors similar to molecule dataset.
             distances - merged molecular distances matrices filled by zero for isolating attention.
             categorical data coded in atoms vector with causal masking in distances equal to 1.

        :param molecules: molecules collection
        :param conditions: conditions collection
        :param max_distance: set distances greater than cutoff to cutoff value
        :param add_cls: add special token at first position
        :param max_neighbors: set neighbors count greater than cutoff to cutoff value
        :param dictionary: predefined conditions to embedding indices mapping
        :param positional_distance: conditions ALIBI-like (but learnable) positional encoding.
            Tokens longer than given value treated as equally far
        :param max_sequence: maximal length of sequence in dataset
        """
        assert len(molecules) == len(conditions), 'reactions and conditions counts mismatch'

        self.molecules = molecules
        self.conditions = conditions
        self.max_distance = max_distance
        self.max_neighbors = max_neighbors
        self.add_cls = add_cls
        self.unpack = unpack
        self.positional_distance = positional_distance

        if dictionary is not None:
            self.dictionary = dictionary
            assert max_sequence, 'max sequence should be provided if an external dictionary is used'
        else:
            self.dictionary = dictionary = {}
            tmp = 0
            for c in conditions:
                if len(c) > tmp:
                    tmp = len(c)
                for x in c:
                    if x not in dictionary:
                        # first 123 reserved for atoms, cls(1), mlm(2), pad(0), sos(121), eos(122)
                        dictionary[x] = len(dictionary) + 123
            if not max_sequence:
                max_sequence = tmp + 1
            else:
                assert max_sequence > tmp, 'given max_sequence less than found in dataset'

        if positional_distance:
            self._mask = clip(tril(arange(max_distance + max_sequence + 3, max_distance + 3, -1).unsqueeze_(0) -
                                   arange(max_sequence, 0, -1).unsqueeze_(1)).to(int32),
                              max=positional_distance + max_distance + 3)
        else:
            self._mask = tril(ones(max_sequence, max_sequence, dtype=int32))

    def __getitem__(self, item: int) -> MoleculeMixerDataPoint:
        dictionary = self.dictionary
        conditions = IntTensor([dictionary[x] for x in self.conditions[item]])
        lc = len(conditions) + 1  # +SOS

        mol = MoleculeDataset(self.molecules, max_distance=self.max_distance, max_neighbors=self.max_neighbors,
                              add_cls=self.add_cls, symmetric_cls=self.add_cls, unpack=self.unpack)[item]

        atoms = cat([mol.atoms, IntTensor([121]), conditions])  # SOS
        causal = cat([mol.atoms, conditions, IntTensor([122])])  # EOS
        neighbors = cat([mol.neighbors, zeros(lc, dtype=int32)])  # disable conditions centrality
        tmp = zeros(len(atoms), len(atoms), dtype=int32)
        tmp[-lc:] = 1  # add unbiased conditions attention to atoms
        tmp[-lc:, -lc:] = self._mask[-lc:, -lc:]  # next token prediction mask for conditions
        tmp[:-lc, :-lc] = mol.distances
        return MoleculeMixerDataPoint(atoms, neighbors, tmp, causal)

    def __len__(self):
        return len(self.molecules)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['MoleculeMixerDataset', 'MoleculeMixerDataPoint', 'MoleculeMixerDataBatch', 'collate_mixed_molecules']
