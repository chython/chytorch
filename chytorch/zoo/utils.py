# -*- coding: utf-8 -*-
#
#  Copyright 2022 Ramil Nugmanov <rnugmano@its.jnj.com>
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
