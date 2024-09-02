# -*- coding: utf-8 -*-
#
# Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from math import inf
from torch import empty, no_grad, addmm, Tensor
from torch.nn import Embedding as tEmbedding, Parameter, init
from torch.nn.functional import embedding
from typing import Optional


class Embedding(tEmbedding):
    """
    LoRA wrapped Embedding layer.
    """
    def __init__(self, *args, neg_inf_idx: Optional[int] = None, **kwargs):
        """
        :param neg_inf_idx: -inf frozen embedding vector

        See torch.nn.Embedding for other params
        """
        super().__init__(*args, **kwargs)
        self.neg_inf_idx = neg_inf_idx
        self.lora_r = 0
        if neg_inf_idx is not None:
            with no_grad():
                self.weight[neg_inf_idx].fill_(-inf)

    def forward(self, x: Tensor) -> Tensor:
        emb = super().forward(x)
        if self.lora_r:
            a = embedding(x, self.lora_a, self.padding_idx, self.max_norm,
                          self.norm_type, self.scale_grad_by_freq, self.sparse)
            return addmm(emb.flatten(end_dim=-2), a.flatten(end_dim=-2), self.lora_b.transpose(0, 1),
                         alpha=self._lora_scaling).view(emb.shape)
        return emb

    def activate_lora(self, lora_r: int = 0, lora_alpha: float = 1.):
        """
        :param lora_r: LoRA factorization dimension
        :param lora_alpha: LoRA scaling factor
        """
        assert lora_r > 0, 'rank should be greater than zero'
        self.weight.requires_grad = False  # freeze main weights
        self.lora_a = Parameter(init.zeros_(empty(self.num_embeddings, lora_r)))
        self.lora_b = Parameter(init.normal_(empty(self.embedding_dim, lora_r)))

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self._lora_scaling = lora_alpha / lora_r

    def merge_lora(self):
        """
        Transform LoRA embedding to normal
        """
        if not self.lora_r:
            return
        self.weight.data += (self.lora_a @ self.lora_b.transpose(0, 1)) * self._lora_scaling
        self.weight.requires_grad = True
        self.lora_r = 0
        del self.lora_a, self.lora_b, self.lora_alpha, self._lora_scaling

    def extra_repr(self) -> str:
        r = super().extra_repr()
        if self.neg_inf_idx is not None:
            r += f', neg_inf_idx={self.neg_inf_idx}'
        if self.lora_r:
            r += f', lora_r={self.lora_r}, lora_alpha={self.lora_alpha}'
        return r


__all__ = ['Embedding']
