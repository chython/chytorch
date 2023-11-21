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
from torch import Tensor
from torch.nn import Dropout, GELU, LayerNorm, Module
from typing import Tuple, Optional, Type
from .attention import GraphormerAttention
from ..lora import Linear


class EncoderLayer(Module):
    r"""EncoderLayer based on torch.nn.TransformerEncoderLayer, but batch always first and returns also attention.

    :param d_model: the number of expected features in the input (required).
    :param nhead: the number of heads in the multiheadattention models (required).
    :param dim_feedforward: the dimension of the feedforward network model (required).
    :param dropout: the dropout value (default=0.1).
    :param activation: the activation function of the intermediate layer. Default: GELU.
    :param layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    :param norm_first: if `True`, layer norm is done prior to self attention, multihead
        attention and feedforward operations, respectively. Otherwise, it's done after.
    :param lora_r: LoRA factorization dimension
    :param lora_alpha: LoRA scaling factor
    :param lora_dropout: LoRA input dropout
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation=GELU, layer_norm_eps=1e-5,
                 norm_first: bool = False, attention: Type[Module] = GraphormerAttention,
                 lora_r: int = 0, lora_alpha: float = 1., lora_dropout: float = 0.):
        super().__init__()
        self.self_attn = attention(d_model, nhead, dropout, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)  # noqa

        self.linear1 = Linear(d_model, dim_feedforward, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear2 = Linear(dim_feedforward, d_model, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.activation = activation()
        self.norm_first = norm_first

    def forward(self, x: Tensor, attn_mask: Optional[Tensor], pad_mask: Optional[Tensor] = None, *,
                cache: Optional[Tuple[Tensor, Tensor]] = None,
                need_embedding: bool = True, need_weights: bool = False) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        nx = self.norm1(x) if self.norm_first else x  # pre-norm or post-norm
        e, a = self.self_attn(nx, attn_mask, pad_mask, cache=cache, need_weights=need_weights)

        if need_embedding:
            x = x + self.dropout1(e)
            if self.norm_first:
                return x + self._ff(self.norm2(x)), a
            # else: post-norm
            x = self.norm1(x)
            return self.norm2(x + self._ff(x)), a
        return None, a

    def merge_lora(self):
        """
        Transform LoRA Encoder to normal
        """
        self.self_attn.merge_lora()
        self.linear1.merge_lora()
        self.linear2.merge_lora()

    def _ff(self, x):
        return self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x)))))


__all__ = ['EncoderLayer']
