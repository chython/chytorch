# -*- coding: utf-8 -*-
#
# Copyright 2022-2024 Ramil Nugmanov <nougmanoff@protonmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from chython import MoleculeContainer
from functools import cached_property
from numpy import empty, ndarray, sqrt, square, ones, digitize, arange, int32
from numpy.random import default_rng
from torch import IntTensor, Size, zeros, ones as t_ones, int32 as t_int32, eye
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map
from torchtyping import TensorType
from typing import Sequence, Tuple, Union, NamedTuple


class ConformerDataPoint(NamedTuple):
    atoms: TensorType['atoms', int]
    hydrogens: TensorType['atoms', int]
    distances: TensorType['atoms', 'atoms', int]


class ConformerDataBatch(NamedTuple):
    atoms: TensorType['batch', 'atoms', int]
    hydrogens: TensorType['batch', 'atoms', int]
    distances: TensorType['batch', 'atoms', 'atoms', int]

    def to(self, *args, **kwargs):
        return ConformerDataBatch(*(x.to(*args, **kwargs) for x in self))

    def cpu(self, *args, **kwargs):
        return ConformerDataBatch(*(x.cpu(*args, **kwargs) for x in self))

    def cuda(self, *args, **kwargs):
        return ConformerDataBatch(*(x.cuda(*args, **kwargs) for x in self))


def collate_conformers(batch, *, collate_fn_map=None) -> ConformerDataBatch:
    """
    Prepares batches of conformers.

    :return: atoms, hydrogens, distances.
    """
    atoms, hydrogens, distances = [], [], []

    for a, h, d in batch:
        atoms.append(a)
        hydrogens.append(h)
        distances.append(d)

    pa = pad_sequence(atoms, True)
    b, s = pa.shape
    tmp = eye(s, dtype=t_int32).repeat(b, 1, 1)  # prevent nan in MHA softmax on padding
    for i, d in enumerate(distances):
        s = d.size(0)
        tmp[i, :s, :s] = d
    return ConformerDataBatch(pa, pad_sequence(hydrogens, True), tmp)


default_collate_fn_map[ConformerDataPoint] = collate_conformers  # add auto_collation to the DataLoader


class ConformerDataset(Dataset):
    def __init__(self, molecules: Sequence[Union[MoleculeContainer, Tuple[ndarray, ndarray, ndarray]]], *,
                 short_cutoff: float = .9, long_cutoff: float = 5., precision: float = .05,
                 add_cls: bool = True, unpack: bool = True, xyz: bool = True, noisy_distance: bool = False):
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

            for i, (n, a) in enumerate(mol.atoms(), self.add_cls):
                atoms[i] = a.atomic_number + 2
                hydrogens[i] = (a.implicit_hydrogens or 0) + 2

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
