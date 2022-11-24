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
from .decoder import *
from .encoder import *
from .fake import *
from .mapping import *
from .permuted import *


# reverse compatibility
ReactionDataset = ReactionEncoderDataset
collate_reactions = collate_encoded_reactions


__all__ = ['ReactionEncoderDataset', 'ReactionDecoderDataset', 'FakeReactionDataset', 'PermutedReactionDataset',
           'MappedReactionDataset',
           'collate_encoded_reactions', 'collate_decoded_reactions', 'collate_faked_reactions',
           'collate_permuted_reactions', 'collate_mapped_reactions',
           # reverse compatibility
           'ReactionDataset', 'collate_reactions']
