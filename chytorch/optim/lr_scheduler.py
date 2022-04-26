# -*- coding: utf-8 -*-
#
#  Copyright 2021, 2022 Ramil Nugmanov <nougmanoff@protonmail.com>
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
