# -*- coding: utf-8 -*-
#
# Copyright 2022, 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from random import random
from torch import IntTensor, Tensor, Size
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Sequence, Union, Optional, Dict


def collate_sequences(batch) -> TensorType['batch', 'atoms', int]:
    """
    Prepares batches of sequences padded by 0
    """
    return pad_sequence(batch, True)


class SMILESTokenizerDataset(Dataset):
    def __init__(self, molecules: Sequence[Union[bytes, MoleculeContainer]], *, format_spec: Optional[str] = None,
                 add_sos: bool = True, add_eos: bool = True,
                 unpack: bool = False, dictionary: Dict[str, int] = None, compressed: bool = True):
        """
        Convert molecules into smiles and tokenize it.

        :param molecules: molecules collection
        :param format_spec: chython smiles formatting params
        :param add_sos: add start token == 1
        :param add_eos: add end token == 2
        :param dictionary: token to idx map. Token idx 0, 1, and 2 should be reserved for PAD SOS EOS.
        :param unpack: unpack molecules
        :param compressed: packed molecules are compressed
        """
        self.molecules = molecules
        self.add_sos = add_sos
        self.add_eos = add_eos
        self.unpack = unpack
        self.dictionary = dictionary if dictionary is not None else {'C': 3}
        self._reverse = None
        self.first_free = max(self.dictionary.values()) + 1
        self.compressed = compressed

        self.format = kwargs = {}
        if format_spec:
            if 'a' in format_spec:
                kwargs['asymmetric_closures'] = True
            if '!s' in format_spec:
                kwargs['stereo'] = False
            if 'A' in format_spec:
                kwargs['aromatic'] = False
            if 'm' in format_spec:
                kwargs['mapping'] = True
            if 'h' in format_spec:
                kwargs['hydrogens'] = True
            if '!b' in format_spec:
                kwargs['bonds'] = False
            if '!z' in format_spec:
                kwargs['charges'] = False
            if 'r' in format_spec:
                kwargs['random'] = True

    def __len__(self):
        return len(self.molecules)

    def size(self, dim):
        if dim == 0:
            return len(self)
        elif dim is None:
            return Size((len(self),))
        raise IndexError

    def __getitem__(self, item: int):
        d = self.dictionary
        m = self.molecules[item]
        if self.unpack:
            m = MoleculeContainer.unpack(m, compressed=self.compressed)

        if self.format.get('random'):
            def w(_):
                return random()
        else:
            w = m._smiles_order(self.format.get('stereo', True))

        smiles = [1] if self.add_sos else []
        for x in m._smiles(w, **self.format):
            if x:
                try:
                    smiles.append(d[x])
                except KeyError:
                    d[x] = self.first_free
                    smiles.append(self.first_free)
                    self.first_free += 1
        if self.add_eos:
            smiles.append(2)
        return IntTensor(smiles)

    def from_tokens(self, tokens):
        d = self.reverse_dictionary
        out = []
        for x in (tokens.tolist() if isinstance(tokens, Tensor) else tokens):
            if x == 0:
                out.append('[PAD]')
            elif x == 1:
                out.append('[SOS]')
            elif x == 2:
                out.append('[EOS]')
            else:
                out.append(d[x])
        return ''.join(out)

    @property
    def reverse_dictionary(self):
        if self._reverse is None or len(self._reverse) != len(self.dictionary):
            self._reverse = {v: k for k, v in self.dictionary.items()}
        return self._reverse


__all__ = ['SMILESTokenizerDataset', 'collate_sequences']
