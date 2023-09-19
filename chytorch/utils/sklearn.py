# -*- coding: utf-8 -*-
#
# Copyright 2023 Ramil Nugmanov <nougmanoff@protonmail.com>
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
