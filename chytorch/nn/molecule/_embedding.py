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
from torch import empty_like
from torch.nn import Module
from ..lora import Embedding


class EmbeddingBag(Module):
    def __init__(self, max_neighbors: int = 14, d_model: int = 1024, perturbation: float = 0., max_tokens: int = 121):
        assert perturbation >= 0, 'zero or positive perturbation expected'
        assert max_tokens >= 121, 'at least 121 tokens should be'
        super().__init__()
        self.atoms_encoder = Embedding(max_tokens, d_model, 0)
        self.neighbors_encoder = Embedding(max_neighbors + 3, d_model, 0)

        self.max_neighbors = max_neighbors
        self.perturbation = perturbation
        self.max_tokens = max_tokens

    def forward(self, atoms, neighbors):
        # cls token in neighbors coded by 0
        x = self.atoms_encoder(atoms) + self.neighbors_encoder(neighbors)

        if self.perturbation and self.training:
            x = x + empty_like(x).uniform_(-self.perturbation, self.perturbation)
        return x


__all__ = ['EmbeddingBag']
