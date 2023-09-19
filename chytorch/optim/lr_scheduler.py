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
from math import cos, pi
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpCosine(_LRScheduler):
    """
    Warm-Up with non-periodic Cosine.

    Note: some code copy-pasted from torch.optim.lr_scheduler.
    """
    def __init__(self, optimizer, decrease_coef=.01, warmup=100, period=1000, last_epoch=-1):
        """
        :param warmup: number of batches/epochs for linear lr warm up.
        :param period: number of batches/epochs for decreasing from lr to decrease_coef * lr.
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.decrease_coef = decrease_coef
        self.warmup = warmup
        self.period = period

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        self._base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self._min_lrs = [lr * decrease_coef for lr in self._base_lrs]
        self._delta_lrs = [(x - y) / 2 for x, y in zip(self._base_lrs, self._min_lrs)]
        self._period_warmup = period + warmup
        self.step()

    def step(self):
        self.last_epoch += 1
        self._last_lr = values = self.get_lr()

        for i, (group, lr) in enumerate(zip(self.optimizer.param_groups, values)):
            group['lr'] = lr

    def get_lr(self):
        if self.last_epoch == 0:
            return [0.] * len(self._base_lrs)
        elif self.last_epoch <= self.warmup:
            return [lr * self.last_epoch / self.warmup for lr in self._base_lrs]
        elif self.last_epoch >= self._period_warmup:
            return self._min_lrs
        c = cos(pi * (self.last_epoch - self.warmup) / self.period) + 1  # [2:0]
        return [x + c * y for x, y in zip(self._min_lrs, self._delta_lrs)]


__all__ = ['WarmUpCosine']
