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
from torch import Tensor, nn
from torch.nn import Dropout, GELU, LayerNorm, Module, SiLU
from typing import Tuple, Optional, Type
from warnings import warn
from .attention import GraphormerAttention
from ..lora import Linear


def _update(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    if prefix + 'linear1.weight' in state_dict:
        warn('fixed chytorch<1.64 checkpoint', DeprecationWarning)
        state_dict[prefix + 'mlp.linear1.weight'] = state_dict.pop(prefix + 'linear1.weight')
        state_dict[prefix + 'mlp.linear2.weight'] = state_dict.pop(prefix + 'linear2.weight')
        if prefix + 'linear1.bias' in state_dict:
            state_dict[prefix + 'mlp.linear1.bias'] = state_dict.pop(prefix + 'linear1.bias')
            state_dict[prefix + 'mlp.linear2.bias'] = state_dict.pop(prefix + 'linear2.bias')


class MLP(Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation=GELU, bias: bool = True):
        super().__init__()
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias)
        self.dropout = Dropout(dropout)

        # ad-hoc for resolving class from name
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        self.activation = activation()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class LLaMAMLP(Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation=SiLU, bias: bool = False):
        super().__init__()
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = Linear(d_model, dim_feedforward, bias=bias)
        self.linear3 = Linear(dim_feedforward, d_model, bias=bias)
        self.dropout = Dropout(dropout)

        # ad-hoc for resolving class from name
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        self.activation = activation()

    def forward(self, x):
        return self.linear3(self.dropout(self.activation(self.linear1(x))) * self.linear2(x))


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
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation=GELU, layer_norm_eps=1e-5,
                 norm_first: bool = False, attention: Type[Module] = GraphormerAttention, mlp: Type[Module] = MLP,
                 norm_layer: Type[Module] = LayerNorm, projection_bias: bool = True, ff_bias: bool = True):
        super().__init__()
        self.self_attn = attention(d_model, nhead, dropout, projection_bias)
        self.mlp = mlp(d_model, dim_feedforward, dropout, activation, ff_bias)

        self.norm1 = norm_layer(d_model, eps=layer_norm_eps)
        self.norm2 = norm_layer(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.norm_first = norm_first
        self._register_load_state_dict_pre_hook(_update)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor], *,
                need_embedding: bool = True, need_weights: bool = False,
                **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        nx = self.norm1(x) if self.norm_first else x  # pre-norm or post-norm
        e, a = self.self_attn(nx, attn_mask, need_weights=need_weights, **kwargs)

        if need_embedding:
            x = x + self.dropout1(e)
            if self.norm_first:
                return x + self.dropout2(self.mlp(self.norm2(x))), a
            # else: post-norm
            x = self.norm1(x)
            return self.norm2(x + self.dropout2(self.mlp(x))), a
        return None, a


__all__ = ['EncoderLayer', 'MLP', 'LLaMAMLP']
