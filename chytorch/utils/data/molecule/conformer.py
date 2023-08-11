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
from chython import MoleculeContainer
from functools import cached_property
from numpy import empty, ndarray, sqrt, square, ones, digitize, arange, int32
from numpy.random import default_rng
from torch import IntTensor, Size, zeros, ones as t_ones, int32 as t_int32
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Tuple, Union
from .._utils import DataTypeMixin, NamedTuple, default_collate_fn_map


class ConformerDataPoint(NamedTuple):
    atoms: TensorType['atoms', int]
    hydrogens: TensorType['atoms', int]
    distances: TensorType['atoms', 'atoms', int]


class ConformerDataBatch(NamedTuple, DataTypeMixin):
    atoms: TensorType['batch', 'atoms', int]
    hydrogens: TensorType['batch', 'atoms', int]
    distances: TensorType['batch', 'atoms', 'atoms', int]


def collate_conformers(batch, *, padding_left: bool = False, collate_fn_map=None) -> ConformerDataBatch:
    """
    Prepares batches of conformers.

    :return: atoms, hydrogens, distances.
    """
    atoms, hydrogens, distances = [], [], []

    for a, h, d in batch:
        if padding_left:
            atoms.append(a.flipud())
            hydrogens.append(h.flipud())
        else:
            atoms.append(a)
            hydrogens.append(h)
        distances.append(d)

    pa = pad_sequence(atoms, True)
    b, s = pa.shape
    tmp = zeros(b, s, s, dtype=t_int32)
    # prevent nan in MHA softmax on padding
    if padding_left:
        tmp[:, :, -1] = 1
    else:
        tmp[:, :, 0] = 1
    for i, d in enumerate(distances):
        s = d.size(0)
        if padding_left:
            tmp[i, -s:, -s:] = d
        else:
            tmp[i, :s, :s] = d
    if padding_left:
        return ConformerDataBatch(pa.fliplr(), pad_sequence(hydrogens, True).fliplr(), tmp)
    return ConformerDataBatch(pa, pad_sequence(hydrogens, True), tmp)


default_collate_fn_map[ConformerDataPoint] = collate_conformers  # add auto_collation to the DataLoader


class ConformerDataset(Dataset):
    def __init__(self, molecules: Sequence[Union[MoleculeContainer, Tuple[ndarray, ndarray, ndarray]]], *,
                 short_cutoff: float =.9, long_cutoff: float = 5., precision: float = .05,
                 add_cls: bool = True, unpack: bool = True, xyz: bool = True,
                 distance_masking_rate: float = 0, noisy_distance: bool = False):
        """
        convert molecules to tuple of:
            atoms vector with atomic numbers + 2,
            vector with implicit hydrogens count shifted by 2,
            matrix with the discretized Euclidian distances between atoms shifted by 3.

        Note: atoms shifted to differentiate from padding equal to zero, special cls token equal to 1,
            and reserved task specific token equal to 2.
            hydrogens shifted to differentiate from padding equal to zero and reserved task-specific token equal to 1.
            distances shifted to differentiate from padding equal to zero, from special distance equal to 1
                that code unreachable atoms/tokens, and self-attention of atoms equal to 2.

        :param molecules: map-like molecules collection or tuples of atomic numbers array,
            hydrogens array, and coordinates/distances array.
        :param short_cutoff: shortest possible distance between atoms
        :param long_cutoff: radius of visible neighbors sphere
        :param precision: discretized segment size
        :param add_cls: add special token at first position
        :param unpack: unpack coordinates from `chython.MoleculeContainer` (True) or use prepared data (False).
            predefined data structure: (vector of atomic numbers, vector of neighbors,
                                        matrix of coordinates or distances).
        :param xyz: provided xyz or distance matrix if unpack=False
        :param distance_masking_rate: probability of masking non-self-loop by 0
        :param noisy_distance: add noise in [-1, 1] range into binarized distance
        """
        if unpack:
            assert xyz, 'xyz should be True if unpack True'
        assert precision > .01 and short_cutoff > .1 and long_cutoff > 1, 'invalid cutoff and precision'
        assert  long_cutoff - short_cutoff > precision, 'precision should be less than cutoff interval'

        self.molecules = molecules
        self.short_cutoff = short_cutoff
        self.long_cutoff = long_cutoff
        self.precision = precision
        self.add_cls = add_cls
        self.unpack = unpack
        self.xyz = xyz
        self.distance_masking_rate = distance_masking_rate
        self.noisy_distance = noisy_distance

        # discrete bins intervals. first 3 bins reserved for shifted coding
        self._bins = arange(short_cutoff - 3 * precision, long_cutoff, precision)
        self._bins[:3] = [-1, 0, .01]  # trick for self-loop coding
        self.max_distance = len(self._bins) - 2  # param for MoleculeEncoder

    def __getitem__(self, item: int) -> ConformerDataPoint:
        mol = self.molecules[item]
        if self.unpack:
            if self.add_cls:
                atoms = t_ones(len(mol) + 1, dtype=t_int32)  # cls token = 1
                hydrogens = zeros(len(mol) + 1, dtype=t_int32)  # cls centrality-encoder disabled by padding trick
            else:
                atoms = IntTensor(len(mol))
                hydrogens = IntTensor(len(mol))

            hgs = mol._hydrogens  # noqa
            for i, (n, a) in enumerate(mol.atoms(), self.add_cls):
                atoms[i] = a.atomic_number + 2
                hydrogens[i] = (hgs[n] or 0) + 2

            xyz = empty((len(mol), 3))
            conformer = mol._conformers[0]  # noqa
            for i, n in enumerate(mol):
                xyz[i] = conformer[n]
        else:
            a, hgs, xyz = mol
            if self.add_cls:
                atoms = t_ones(len(a) + 1, dtype=t_int32)
                hydrogens = zeros(len(a) + 1, dtype=t_int32)

                atoms[1:] = IntTensor(a + 2)
                hydrogens[1:] = IntTensor(hgs + 2)
            else:
                atoms = IntTensor(a + 2)
                hydrogens = IntTensor(hgs + 2)

        if self.xyz:
            diff = xyz[None, :, :] - xyz[:, None, :]  # NxNx3
            dist = sqrt(square(diff).sum(axis=-1))  # NxN
        else:
            dist = xyz

        dist = digitize(dist, self._bins)
        if self.noisy_distance:
            dist += (dist > 2) * self.generator.integers(-1, 1, size=dist.shape, endpoint=True)
        if self.distance_masking_rate:
            dist *= ((dist <= 2) | (self.generator.random(dist.shape) > self.distance_masking_rate))
        if self.add_cls:  # set cls to atoms distance equal to 0
            tmp = ones((len(atoms), len(atoms)), dtype=int32)
            tmp[1:, 1:] = dist
            dist = tmp
        return ConformerDataPoint(atoms, hydrogens, IntTensor(dist))

    def __len__(self):
        return len(self.molecules)

    def size(self, dim):
        if dim == 0:
            return len(self.molecules)
        elif dim is None:
            return Size((len(self),))
        raise IndexError

    @cached_property
    def generator(self):
        return default_rng()


__all__ = ['ConformerDataset', 'ConformerDataPoint', 'ConformerDataBatch', 'collate_conformers']
