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
from functools import lru_cache
from torch import ones, zeros


@lru_cache()
def k_fold_mask(k_fold, ensemble, batch_size, train, device=None):
    assert k_fold >= 3, 'k-fold should be at least 3'
    assert not ensemble % k_fold, 'ensemble should be divisible by k-fold'
    assert not batch_size % k_fold, 'batch size should be divisible by k-fold'

    if train:
        m = ones(batch_size, ensemble, device=device)  # k-th fold mask
        disable = 0.
    else:  # test/validation
        m = zeros(batch_size, ensemble, device=device)  # k-th fold mask
        disable = 1.

    batch_size //= k_fold
    ensemble //= k_fold
    for n in range(k_fold):  # disable folds
        m[n * batch_size: n * batch_size + batch_size, n * ensemble: n * ensemble + ensemble] = disable
    return m


__all__ = ['k_fold_mask']
