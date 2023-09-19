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
from functools import lru_cache
from torch import ones, zeros


@lru_cache()
def k_fold_mask(k_fold, ensemble, batch_size, train, device=None):
    """
    :param k_fold: number of folds
    :param ensemble: number of predicting heads
    :param batch_size: size of batch
    :param train: create train of test mask. train - mask only 1/k_fold of data. test - mask 4/k_fold of data.

    :param device: device of mask
    """
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
