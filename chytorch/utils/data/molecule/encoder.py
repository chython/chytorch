# -*- coding: utf-8 -*-
#
# Copyright 2021-2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from numpy import minimum, nan_to_num, ones
from scipy.sparse.csgraph import shortest_path
from torch import IntTensor, Size, int32, ones as t_ones, zeros, eye
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Union, NamedTuple
from zlib import decompress
from .._abc import default_collate_fn_map


class MoleculeDataPoint(NamedTuple):
    atoms: TensorType['atoms', int]
    neighbors: TensorType['atoms', int]
    distances: TensorType['atoms', 'atoms', int]


class MoleculeDataBatch(NamedTuple):
    atoms: TensorType['batch', 'atoms', int]
    neighbors: TensorType['batch', 'atoms', int]
    distances: TensorType['batch', 'atoms', 'atoms', int]

    def to(self, *args, **kwargs):
        return MoleculeDataBatch(*(x.to(*args, **kwargs) for x in self))

    def cpu(self, *args, **kwargs):
        return MoleculeDataBatch(*(x.cpu(*args, **kwargs) for x in self))

    def cuda(self, *args, **kwargs):
        return MoleculeDataBatch(*(x.cuda(*args, **kwargs) for x in self))


def collate_molecules(batch, *, padding_left: bool = False, collate_fn_map=None) -> MoleculeDataBatch:
    """
    Prepares batches of molecules.

    :return: atoms, neighbors, distances.
    """
    atoms, neighbors, distances = [], [], []

    for a, n, d in batch:
        if padding_left:
            atoms.append(a.flipud())
            neighbors.append(n.flipud())
        else:
            atoms.append(a)
            neighbors.append(n)
        distances.append(d)

    pa = pad_sequence(atoms, True)
    b, s = pa.shape
    tmp = eye(s, dtype=int32).repeat(b, 1, 1)  # prevent nan in MHA softmax on padding
    for i, d in enumerate(distances):
        s = d.size(0)
        if padding_left:
            tmp[i, -s:, -s:] = d
        else:
            tmp[i, :s, :s] = d
    if padding_left:
        return MoleculeDataBatch(pa.fliplr(), pad_sequence(neighbors, True).fliplr(), tmp)
    return MoleculeDataBatch(pa, pad_sequence(neighbors, True), tmp)


default_collate_fn_map[MoleculeDataPoint] = collate_molecules  # add auto_collation to the DataLoader


class MoleculeDataset(Dataset):
    def __init__(self, molecules: Sequence[Union[MoleculeContainer, bytes]], *, max_distance: int = 10,
                 add_cls: bool = True, max_neighbors: int = 14, unpack: bool = False, distance_cutoff=None,
                 compressed: bool = True):
        """
        convert molecules to tuple of:
            atoms vector with atomic numbers + 2,
            neighbors vector with connected neighbored atoms count including implicit hydrogens count shifted by 2,
            distance matrix with the shortest paths between atoms shifted by 2.

        Note: atoms shifted to differentiate from padding equal to zero, special cls token equal to 1, and reserved MLM
              task token equal to 2.
              neighbors shifted to differentiate from padding equal to zero and reserved MLM task token equal to 1.
              distances shifted to differentiate from padding equal to zero and from special distance equal to 1
              that code unreachable atoms (e.g. salts).

        :param molecules: molecules collection
        :param max_distance: set distances greater than cutoff to cutoff value
        :param add_cls: add special token at first position
        :param max_neighbors: set neighbors count greater than cutoff to cutoff value
        :param unpack: unpack molecules
        :param compressed: packed molecules are compressed
        """
        self.molecules = molecules
        # distance_cutoff is deprecated
        self.max_distance = distance_cutoff if distance_cutoff is not None else max_distance
        self.add_cls = add_cls
        self.max_neighbors = max_neighbors
        self.unpack = unpack
        self.compressed = compressed

    def __getitem__(self, item: int) -> MoleculeDataPoint:
        mol = self.molecules[item]
        if self.unpack:
            try:
                from ._unpack import unpack
            except ImportError:  # windows?
                mol = MoleculeContainer.unpack(mol, compressed=self.compressed)
            else:
                if self.compressed:
                    mol = decompress(mol)
                atoms, neighbors, distances, _ = unpack(mol, self.add_cls, self.max_neighbors, self.max_distance)
                return MoleculeDataPoint(IntTensor(atoms), IntTensor(neighbors), IntTensor(distances))

        nc = self.max_neighbors
        if self.add_cls:
            atoms = t_ones(len(mol) + 1, dtype=int32)  # cls token = 1
            neighbors = zeros(len(mol) + 1, dtype=int32)  # cls centrality-encoder disabled by padding trick
        else:
            atoms = IntTensor(len(mol))
            neighbors = IntTensor(len(mol))

        ngb = mol._bonds  # noqa speedup
        hgs = mol._hydrogens  # noqa
        for i, (n, a) in enumerate(mol.atoms(), self.add_cls):
            atoms[i] = a.atomic_number + 2
            nb = len(ngb[n]) + (hgs[n] or 0)  # treat bad valence as 0-hydrogen
            if nb > nc:
                nb = nc
            neighbors[i] = nb + 2

        sp = shortest_path(mol.adjacency_matrix(), method='FW', directed=False, unweighted=True) + 2
        nan_to_num(sp, copy=False, posinf=1)
        minimum(sp, self.max_distance + 2, out=sp)

        if self.add_cls:
            tmp = ones((len(atoms), len(atoms)))
            tmp[1:, 1:] = sp
            sp = tmp
        return MoleculeDataPoint(atoms, neighbors, IntTensor(sp))

    def __len__(self):
        return len(self.molecules)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['MoleculeDataset', 'MoleculeDataPoint', 'MoleculeDataBatch', 'collate_molecules']
