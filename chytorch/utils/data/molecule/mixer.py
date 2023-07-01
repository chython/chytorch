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
                 *, max_distance: int = 10, max_neighbors: int = 14, add_cls: bool = False, unpack: bool = False,
                 dictionary: Dict[Hashable, int] = None, positional_distance: int = 0, masking_rate: float = 0,
                 max_tokens: Optional[int] = None, auto_eos: bool = True):
        """
        convert molecules and categorical conditions to tuple of:
            atoms, neighbors and distances tensors similar to molecule dataset.
             distances - merged molecular distances matrices filled by zero for isolating attention.
             categorical data coded in atoms vector with causal masking in distances equal to 1.

        Note: Dataset automatically adds SOS token.

        :param molecules: molecules collection
        :param conditions: conditions collection
        :param max_distance: set distances greater than cutoff to cutoff value
        :param add_cls: add special token at first position
        :param max_neighbors: set neighbors count greater than cutoff to cutoff value
        :param dictionary: predefined tokens to embedding indices mapping
        :param positional_distance: conditions ALIBI-like (but learnable) positional encoding.
            Tokens longer than given value treated as equally far. Disabled by default.
        :param masking_rate: probability of masking non-self-loop by 0
        :param max_tokens: maximal length of sequence in dataset including SOS and EOS or other special tokens
        :param auto_eos: automatically add EOS token.
        """
        assert len(molecules) == len(conditions), 'reactions and conditions counts mismatch'

        self.molecules = molecules
        self.conditions = conditions
        self.positional_distance = positional_distance
        self.auto_eos = auto_eos

        self._dataset = MoleculeDataset(molecules, max_distance=max_distance, max_neighbors=max_neighbors,
                                        add_cls=add_cls, unpack=unpack, masking_rate=masking_rate)

        if dictionary is not None:
            self.dictionary = dictionary
            assert max_tokens, 'max_tokens should be provided if an external dictionary is used'
        else:
            self.dictionary = dictionary = {}
            tmp = 0
            for c in conditions:
                if len(c) > tmp:
                    tmp = len(c)
                for x in c:
                    if x not in dictionary:
                        # first 122 reserved for atoms, cls(1), mlm(2), pad(0), sos(121), if auto-eos: eos(122)
                        dictionary[x] = len(dictionary) + 122 + auto_eos
            if not max_tokens:
                max_tokens = tmp + 1 + auto_eos  # +SOS+EOS?
            else:
                assert max_tokens >= tmp + 1 + auto_eos, 'given max_tokens less than found in dataset'

        self.max_tokens = max_tokens
        if positional_distance:
            assert 1 < positional_distance <= max_tokens, 'positional_distance should in [2, max_tokens] range'
            self._mask = clip(tril(arange(max_distance + max_tokens + 2, max_distance + 2, -1).unsqueeze_(0) -
                                   arange(max_tokens, 0, -1).unsqueeze_(1)).to(int32),
                              max=positional_distance + max_distance + 1).fill_diagonal_(1)
        else:
            self._mask = tril(ones(max_tokens, max_tokens, dtype=int32))

    def __getitem__(self, item: int) -> MoleculeMixerDataPoint:
        dictionary = self.dictionary
        conditions = [dictionary[x] for x in self.conditions[item]]
        if not conditions and not self.auto_eos:
            raise ValueError('empty conditions without auto_eos')

        mol = self._dataset[item]
        size = len(mol.atoms)
        c = len(conditions) + self.auto_eos
        sc = size + c
        neighbors = cat([mol.neighbors, zeros(c, dtype=int32)])

        # fill distance matrix
        tmp = zeros(sc, sc, dtype=int32)
        tmp[:size, :size] = mol.distances
        tmp[size:, :size] = 1  # add conditions to atoms attention
        tmp[size:, size:] = self._mask[-c:, -c:]  # next token prediction mask for conditions

        conditions = IntTensor(conditions)
        if self.auto_eos:
            atoms = cat([mol.atoms, IntTensor([121]), conditions])  # SOS
            causal = cat([mol.atoms, conditions, IntTensor([122])])  # EOS
        else:
            atoms = cat([mol.atoms, IntTensor([121]), conditions[:-1]])  # SOS
            causal = cat([mol.atoms, conditions])
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
