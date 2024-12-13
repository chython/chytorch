# -*- coding: utf-8 -*-
#
# Copyright 2021-2024 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from functools import partial
from numpy import minimum, nan_to_num
from scipy.sparse.csgraph import shortest_path
from torch import IntTensor, Size, int32, ones, zeros, eye, empty, triu, Tensor, cat
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map
from torchtyping import TensorType
from typing import Sequence, Union, NamedTuple, Tuple
from zlib import decompress


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


left_padded_collate_molecules = partial(collate_molecules, padding_left=True)
default_collate_fn_map[MoleculeDataPoint] = collate_molecules  # add auto_collation to the DataLoader


class MoleculeDataset(Dataset):
    def __init__(self, molecules: Sequence[Union[MoleculeContainer, bytes]], *,
                 add_cls: bool = True, cls_token: Union[int, Tuple[int, ...], Sequence[int], Sequence[Tuple[int, ...]],
                     TensorType['cls', int], TensorType['dataset', 1, int], TensorType['dataset', 'cls', int]] = 1,
                 max_distance: int = 10, max_neighbors: int = 14,
                 attention_schema: str = 'bert', components_attention: bool = True,
                 unpack: bool = False, compressed: bool = True, distance_cutoff=None):
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
        :param attention_schema: attention between CLS and atoms:
            bert - symmetrical without masks;
            causal - masked from atoms to cls, causal between cls multi-prompts (triangle mask) and full to atoms;
            directed - masked from atoms to cls and between cls, but full to atoms (self-attention of cls is kept);
        :param components_attention: enable or disable attention between subgraphs
        :param unpack: unpack molecules
        :param compressed: packed molecules are compressed
        :param cls_token: idx of cls token (int), or multiple tokens for multi-prompt (tuple, 1d-tensor),
            or individual token per sample (Sequence, column-vector) or
            individual multi-prompt per sample (Sequence of tuples, 2d-tensor).
        """
        if not isinstance(cls_token, int) or cls_token != 1:
            assert add_cls, 'non-default value of cls_token requires add_cls=True'
        assert attention_schema in ('bert', 'causal', 'directed'), 'Invalid attention schema'

        self.molecules = molecules
        # distance_cutoff is deprecated
        self.max_distance = distance_cutoff if distance_cutoff is not None else max_distance
        self.add_cls = add_cls
        self.max_neighbors = max_neighbors
        self.unpack = unpack
        self.compressed = compressed
        self.cls_token = cls_token
        self.attention_schema = attention_schema
        self.components_attention = components_attention

    def __getitem__(self, item: int) -> MoleculeDataPoint:
        mol = self.molecules[item]

        # cls setup lookup
        cls_token = self.cls_token
        if not self.add_cls:
            cls_cnt = 0
        elif isinstance(cls_token, int):
            cls_cnt = 1
        elif isinstance(cls_token, tuple):
            cls_cnt = len(cls_token)
            assert cls_cnt > 1, 'wrong multi-prompt setup'
            cls_token = IntTensor(cls_token)
        elif isinstance(cls_token, Sequence):
            if isinstance(ct := cls_token[item], int):
                assert isinstance(cls_token[0], int), 'inconsistent cls_token data'
                cls_cnt = 1
                cls_token = ct
            elif isinstance(ct, Sequence):
                cls_cnt = len(ct)
                assert cls_cnt > 1, 'wrong multi-prompt setup'
                assert isinstance(cls_token[0], Sequence) and cls_cnt == len(cls_token[0]), 'inconsistent cls_token data'
                cls_token = IntTensor(ct)
            else:
                raise TypeError('cls_token must be int, tuple of ints, sequence of ints or tuples of ints or 1,2-d tensor')
        elif isinstance(cls_token, Tensor):
            if cls_token.dim() == 1:
                cls_cnt = cls_token.size(0)
                assert cls_cnt > 1, 'wrong multi-prompt setup'
            elif cls_token.dim() == 2:
                cls_cnt = cls_token.size(1)
                cls_token = cls_token[item]
            else:
                raise TypeError('cls_token must be int, tuple of ints, sequence of ints or tuples of ints or 1,2-d tensor')
        else:
            raise TypeError('cls_token must be int, tuple of ints, sequence of ints or tuples of ints or 1,2-d tensor')

        if self.unpack:
            try:
                from ._unpack import unpack
            except ImportError:  # windows?
                mol = MoleculeContainer.unpack(mol, compressed=self.compressed)
            else:
                if self.compressed:
                    mol = decompress(mol)
                atoms, neighbors, distances, _ = unpack(mol, cls_cnt == 1,  # only single cls token supported by cython ext
                                                        # causal and directed have the same mask for 1 cls token case
                                                        self.attention_schema == 'bert',
                                                        self.components_attention, self.max_neighbors,
                                                        self.max_distance)
                atoms = IntTensor(atoms)
                neighbors = IntTensor(neighbors)
                distances = IntTensor(distances)
                if cls_cnt == 1:
                    # token already pre-allocated
                    if isinstance(cls_token, Tensor) or cls_token != 1:
                        # change default value (1)
                        atoms[0] = cls_token
                elif cls_cnt:  # expand atoms with cls tokens
                    atoms = cat([cls_token, atoms])
                    neighbors = cat([zeros(cls_cnt, dtype=int32), neighbors])
                    distances = self._add_cls_to_distances(distances, cls_cnt)
                return MoleculeDataPoint(atoms, neighbors, distances)

        token_cnt = len(mol) + cls_cnt
        atoms = empty(token_cnt, dtype=int32)
        neighbors = zeros(token_cnt, dtype=int32)  # cls centrality-encoder disabled by padding trick

        nc = self.max_neighbors
        ngb = mol._bonds  # noqa speedup
        for i, (n, a) in enumerate(mol.atoms(), cls_cnt):
            atoms[i] = a.atomic_number + 2
            nb = len(ngb[n]) + (a.implicit_hydrogens or 0)  # treat bad valence as 0-hydrogen
            if nb > nc:
                nb = nc
            neighbors[i] = nb + 2

        distances = shortest_path(mol.adjacency_matrix(), method='FW', directed=False, unweighted=True) + 2
        nan_to_num(distances, copy=False, posinf=self.components_attention)
        minimum(distances, self.max_distance + 2, out=distances)
        distances = IntTensor(distances)

        if cls_cnt:
            atoms[:cls_cnt] = cls_token
            distances = self._add_cls_to_distances(distances, cls_cnt)
        return MoleculeDataPoint(atoms, neighbors, distances)

    def __len__(self):
        return len(self.molecules)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError

    def _add_cls_to_distances(self, distances, cls_cnt):
        total = distances.size(0) + cls_cnt
        if self.attention_schema == 'bert':  # everything to everything
            tmp = ones(total, total, dtype=int32)
        elif self.attention_schema == 'causal':
            tmp = triu(ones(total, total, dtype=int32))
        else:  # CLS to atoms but not back
            tmp = eye(total, dtype=int32)  # self attention of cls tokens
            tmp[:cls_cnt, cls_cnt:] = 1  # cls to atom attention
        tmp[cls_cnt:, cls_cnt:] = distances
        return tmp


__all__ = ['MoleculeDataset', 'MoleculeDataPoint', 'MoleculeDataBatch',
           'collate_molecules', 'left_padded_collate_molecules']
