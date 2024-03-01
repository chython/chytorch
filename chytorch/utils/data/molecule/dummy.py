# -*- coding: utf-8 -*-
#
# Copyright 2024 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from chython import smiles
from .encoder import MoleculeDataset


def thiacalix_n_arene_dataset(n=4, size=10_000, **kwargs):
    """
    Create a dummy dataset for testing purposes with thiacalix[n]arenes.

    :param n: number of macrocycle fragments. Each fragment contains 12 atoms.
    :param size: dataset size
    :param kwargs: other params of MoleculeDataset
    """
    assert n >= 3, 'n must be greater than 3'
    prefix = 'C12=CC(C(C)(C)C)=CC(=C2O)S'
    postfix = 'C2=CC(C(C)(C)C)=CC(=C2O)S1'
    chain = ''.join('C2=CC(C(C)(C)C)=CC(=C2O)S' for _ in range(n - 2))

    return MoleculeDataset([smiles(prefix + chain + postfix)] * size, **kwargs)


__all__ = ['thiacalix_n_arene_dataset']
