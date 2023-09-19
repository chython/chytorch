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
from inspect import getfullargspec, isclass, isfunction


def pass_suitable_args(cls, kwargs, remap=None):
    """
    Pass to the function or class only suitable arguments.

    :param cls:
    :param kwargs: any kw arguments
    :param remap: rename arguments before passing
    """
    if isclass(cls):
        f = cls.__init__
    elif isfunction(cls):
        f = cls
    else:
        raise TypeError

    if remap is None:
        remap = {}
    else:
        remap = {v: k for k, v in remap.items()}

    kw = {remap.get(k, k): k for k in getfullargspec(f).args + getfullargspec(f).kwonlyargs}
    return cls(**{kw[k]: v for k, v in kwargs.items() if k in kw and k not in remap})


__all__ = ['pass_suitable_args']
