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
from math import inf
from torch import diag, bool as t_bool, logsumexp, eye
from torch.nn.functional import cosine_similarity, normalize
from torchtyping import TensorType


def contrastive_loss(x: TensorType['batch', 'embedding'], temperature: float = .5):
    """
    Contrastive loss for embeddings. Batch should contain even number of elements.
    First B/2 elements are first elements of positive pairs. Second > second.
    """
    b = x.size(0)
    x = normalize(x)  # euclidian norm
    sim = cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1) / temperature  # Bx1xE @ 1xBxE > BxBxE > BxB
    sim.masked_fill_(eye(b, dtype=t_bool, device=sim.device), -inf)  # mask self-similarity
    return logsumexp(sim, dim=-1).mean() - 2 * diag(sim, b // 2).mean()


__all__ = ['contrastive_loss']
