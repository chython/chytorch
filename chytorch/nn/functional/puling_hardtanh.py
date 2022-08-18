# -*- coding: utf-8 -*-
#
#  Copyright 2022 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of chytorch.
#
#  chytorch is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from torch import clamp
from torch.autograd import Function


class _PulingHardtanh(Function):
    """
    Returns same values as Hardtanh, but keeps gradients of out of range values directed to range zone.
    Possible to set only top or bot  border like torch.clamp.
    """

    @staticmethod
    def forward(ctx, input, mn, mx):
        ctx.save_for_backward(input)
        ctx.clamp_mn = mn
        ctx.clamp_mx = mx
        return clamp(input, mn, mx)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        mn, mx = ctx.clamp_mn, ctx.clamp_mx

        if mn is not None:
            c = input > mn
            grad_output = c * grad_output + clamp(~c * grad_output, None, 0)
        if mx is not None:
            c = input < mx
            grad_output = c * grad_output + clamp(~c * grad_output, 0, None)
        return grad_output, None, None


puling_hardtanh = _PulingHardtanh.apply


__all__ = ['puling_hardtanh']
