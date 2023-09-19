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
