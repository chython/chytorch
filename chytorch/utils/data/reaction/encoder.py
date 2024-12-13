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
from chython import ReactionContainer
from itertools import chain, repeat
from torch import IntTensor, cat, zeros, int32, Size, eye
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map
from torchtyping import TensorType
from typing import Sequence, Union, NamedTuple
from ..molecule import MoleculeDataset


class ReactionEncoderDataPoint(NamedTuple):
    atoms: TensorType['atoms', int]
    neighbors: TensorType['atoms', int]
    distances: TensorType['atoms', 'atoms', int]
    roles: TensorType['atoms', int]


class ReactionEncoderDataBatch(NamedTuple):
    atoms: TensorType['batch', 'atoms', int]
    neighbors: TensorType['batch', 'atoms', int]
    distances: TensorType['batch', 'atoms', 'atoms', int]
    roles: TensorType['batch', 'atoms', int]

    def to(self, *args, **kwargs):
        return ReactionEncoderDataBatch(*(x.to(*args, **kwargs) for x in self))

    def cpu(self, *args, **kwargs):
        return ReactionEncoderDataBatch(*(x.cpu(*args, **kwargs) for x in self))

    def cuda(self, *args, **kwargs):
        return ReactionEncoderDataBatch(*(x.cuda(*args, **kwargs) for x in self))


def collate_encoded_reactions(batch, *, collate_fn_map=None) -> ReactionEncoderDataBatch:
    """
    Prepares batches of reactions.

    :return: atoms, neighbors, distances, atoms roles.
    """
    atoms, neighbors, distances, roles = [], [], [], []
    for a, n, d, r in batch:
        atoms.append(a)
        neighbors.append(n)
        roles.append(r)
        distances.append(d)

    pa = pad_sequence(atoms, True)
    b, s = pa.shape
    tmp = eye(s, dtype=int32).repeat(b, 1, 1)  # prevent nan in MHA softmax on padding
    for n, d in enumerate(distances):
        s = d.size(0)
        tmp[n, :s, :s] = d
    return ReactionEncoderDataBatch(pa, pad_sequence(neighbors, True), tmp, pad_sequence(roles, True))


default_collate_fn_map[ReactionEncoderDataPoint] = collate_encoded_reactions  # add auto_collation to the DataLoader


class ReactionEncoderDataset(Dataset):
    def __init__(self, reactions: Sequence[Union[ReactionContainer, bytes]], *, max_distance: int = 10,
                 max_neighbors: int = 14, add_cls: bool = True, add_molecule_cls: bool = True,
                 hide_molecule_cls: bool = True, unpack: bool = False, distance_cutoff=None, compressed: bool = True):
        """
        convert reactions to tuple of:
            atoms, neighbors and distances tensors similar to molecule dataset.
             distances - merged molecular distances matrices filled by zero for isolating attention.
            roles: 2 reactants, 3 products, 0 padding, 1 cls token.

        :param reactions: reactions collection
        :param max_distance: set distances greater than cutoff to cutoff value
        :param add_cls: add special token at first position
        :param add_molecule_cls: add special token at first position of each molecule
        :param hide_molecule_cls: disable molecule cls in reaction lvl (mark as padding)
        :param max_neighbors: set neighbors count greater than cutoff to cutoff value
        :param unpack: unpack reactions
        :param compressed: packed reactions are compressed
        """
        if not add_molecule_cls:
            assert not hide_molecule_cls, 'add_molecule_cls should be True if hide_molecule_cls is True'
        self.reactions = reactions
        # distance_cutoff is deprecated
        self.max_distance = distance_cutoff if distance_cutoff is not None else max_distance
        self.add_cls = add_cls
        self.add_molecule_cls = add_molecule_cls
        self.hide_molecule_cls = hide_molecule_cls
        self.max_neighbors = max_neighbors
        self.unpack = unpack
        self.compressed = compressed

    def __getitem__(self, item: int) -> ReactionEncoderDataPoint:
        rxn = self.reactions[item]
        if self.unpack:
            rxn = ReactionContainer.unpack(rxn, compressed=self.compressed)
        molecules = MoleculeDataset(rxn.reactants + rxn.products, max_distance=self.max_distance,
                                    max_neighbors=self.max_neighbors, add_cls=self.add_molecule_cls)

        if self.add_cls:
            # disable rxn cls in molecules encoder
            atoms, neighbors, roles = [IntTensor([0])], [IntTensor([0])], [1]
        else:
            atoms, neighbors, roles = [], [], []
        distances = []
        for i, (m, r) in enumerate(chain(zip(rxn.reactants, repeat(2)), zip(rxn.products, repeat(3)))):
            a, n, d = molecules[i]
            atoms.append(a)
            neighbors.append(n)
            distances.append(d)
            if self.add_molecule_cls:
                # (dis|en)able molecule cls in reaction encoder
                roles.append(0 if self.hide_molecule_cls else r)
            roles.extend(repeat(r, len(m)))

        tmp = zeros(len(roles), len(roles), dtype=int32)
        if self.add_cls:
            tmp[0, 0] = 1  # prevent nan in MHA softmax.
            i = 1
        else:
            i = 0
        for d in distances:
            j = i + d.size(0)
            tmp[i:j, i:j] = d
            i = j
        return ReactionEncoderDataPoint(cat(atoms), cat(neighbors), tmp, IntTensor(roles))

    def __len__(self):
        return len(self.reactions)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError


__all__ = ['ReactionEncoderDataset', 'ReactionEncoderDataPoint', 'ReactionEncoderDataBatch',
           'collate_encoded_reactions']
