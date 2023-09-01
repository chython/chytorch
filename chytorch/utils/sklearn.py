# -*- coding: utf-8 -*-
#
#  Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
from torch import no_grad
from torch.nn import Module
from torch.utils.data import Dataset
from typing import Type


class DatasetTransformer:
    """
    Wrap Datasets to provide sklearn transformer API for pipeline building.

        from chytorch.utils.data import SMILESDataset, MoleculeDataset, collate_molecules
        from functools import partial
        from sklearn.pipeline import Pipeline

        sw = DatasetTransformer(SMILESDataset)  # smiles to molecule converter
        mw = DatasetTransformer(MoleculeDataset, collate_fn=partial(collate_molecules, padding_left=True)) #
        p = Pipeline([('smiles', sw), ('molecule', mw)])
        p.transform(['CN(C)C', 'CCO'])
    """
    def __init__(self, dataset: Type[Dataset], *args, collate_fn=None, **kwargs):
        """
        :param dataset: class to wrap
        :param collate_fn: apply batch collate function to the output of dataset. disabled by default
        :param args kwargs: hyperparameters for dataset
        """
        self.dataset = dataset
        self.args = args
        self.kwargs = kwargs
        self.collate_fn = collate_fn

    def fit(self, X, y=None):
        """
        Do nothing. Sklearn API
        """
        return self

    def fit_transform(self, X, y=None):
        """
        Transform data.
        """
        return self.transform(X)

    def transform(self, X):
        """
        Transform data.
        """
        ds = self.dataset(X, *self.args, **self.kwargs)
        batch = list(ds)
        if self.collate_fn is not None:
            return self.collate_fn(batch)
        return batch


class ModelEstimatorTransformer:
    """
    Wrap Neural network to provide sklearn estimator API for pipeline building.

        from sklearn.pipeline import Pipeline
        from chytorch.nn import MoleculeEncoder

        net = MoleculeEncoder()
        sm = ModelEstimatorTransformer(net)
        p = Pipeline([('smiles', sw), ('molecule', mw), ('model', sm)])
        p.predict(['CN(C)C', 'CCO'])
    """
    def __init__(self, model: Module, *args, **kwargs):
        """
        :param model: pytorch model
        :param args kwargs: params for proper model initialization from pickle dump
        """
        self.model = model.eval()
        self.args = args
        self.kwargs = kwargs

    @no_grad()
    def predict(self, X):
        return self.model(X)

    @no_grad()
    def transform(self, X):
        return self.model(X)

    def fit(self, X, y):
        """
        Do nothing. Sklearn API
        """
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def __getstate__(self):
        return {'model': type(self.model), 'args': self.args, 'kwargs': self.kwargs, 'weights': self.model.state_dict()}

    def __setstate__(self, state):
        self.model = state['model'](*state['args'], **state['kwargs']).eval()
        self.model.load_state_dict(state['weights'])
        self.args = state['args']
        self.kwargs = state['kwargs']


__all__ = ['DatasetTransformer', 'ModelEstimatorTransformer']
