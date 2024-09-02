# -*- coding: utf-8 -*-
#
# Copyright 2023, 2024 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from math import sqrt
from torch import empty, addmm, Tensor
from torch.nn import Linear as tLinear, Parameter, init
from torch.nn.functional import dropout


class Linear(tLinear):
    """
    LoRA wrapped Linear layer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_r = 0

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        if self.lora_r:
            if self.training and self.lora_dropout:
                x = dropout(x, self.lora_dropout)
            a = x @ self.lora_a.transpose(0, 1)
            return addmm(out.flatten(end_dim=-2), a.flatten(end_dim=-2), self.lora_b.transpose(0, 1),
                         alpha=self._lora_scaling).view(out.shape)
        return out

    def activate_lora(self, lora_r: int = 0, lora_alpha: float = 1., lora_dropout: float = 0.):
        """
        :param lora_r: LoRA factorization dimension
        :param lora_alpha: LoRA scaling factor
        :param lora_dropout: LoRA input dropout
        """
        assert lora_r > 0, 'rank should be greater than zero'
        self.weight.requires_grad = False  # freeze main weights
        self.lora_a = Parameter(init.kaiming_uniform_(empty(lora_r, self.in_features), a=sqrt(5)))
        self.lora_b = Parameter(init.zeros_(empty(self.out_features, lora_r)))

        self.lora_r = lora_r
        self.lora_dropout = lora_dropout
        self.lora_alpha = lora_alpha
        self._lora_scaling = lora_alpha / lora_r

    def merge_lora(self):
        """
        Transform LoRA linear to normal
        """
        if not self.lora_r:
            return
        self.weight.data += (self.lora_b @ self.lora_a) * self._lora_scaling
        self.weight.requires_grad = True
        self.lora_r = 0
        del self.lora_a, self.lora_b, self.lora_dropout, self.lora_alpha, self._lora_scaling

    def extra_repr(self) -> str:
        r = super().extra_repr()
        if self.lora_r:
            return r + f', lora_r={self.lora_r}, lora_alpha={self.lora_alpha}, lora_dropout={self.lora_dropout}'
        return r


__all__ = ['Linear']
