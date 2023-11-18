<img src="https://github.com/chython/chytorch/assets/2013465/2985c43e-5b00-4ae4-9130-c01e28f6bdd9" width="100" height="100"> Chytorch [kʌɪtɔːrtʃ]
====================


Library for modeling molecules and reactions in torch way.

Installation
------------

Use `pip install chytorch` to install release version.

Or `pip install .` in source code directory to install DEV version.

Pretrained models
-----------------

Chytorch main package doesn't include models zoo.
Each model has its own named package and can be installed separately.
Installed models can be imported as `from chytorch.zoo.<model_name> import Model`.


Usage
-----

`chytorch.nn.MoleculeEncoder` and `chytorch.nn.ReactionEncoder` - core graphormer layers for molecules and reactions.
API is combination of `torch.nn.TransformerEncoderLayer` with `torch.nn.TransformerEncoder`. 

**Batch preparation:**

`chytorch.utils.data.MoleculeDataset` and `chytorch.utils.data.ReactionDataset` - Map-like on-the-fly dataset generators for molecules and reactions.
Supported `chython.MoleculeContainer` and `chython.ReactionContainer` objects, and bytes-packed structures.

`chytorch.utils.data.collate_molecules` and `chytorch.utils.data.collate_reactions` - collate functions for `torch.utils.data.DataLoader`.

Note: torch DataLoader automatically do proper collation since 1.13 release.

Example:

    from chytorch.utils.data import MoleculeDataset, SMILESDataset
    from torch.utils.data import DataLoader

    data = ['CCO', 'CC=O']
    ds = MoleculeDataset(SMILESDataset(data, cache={}))
    dl = DataLoader(ds, batch_size=10)

**Forward call:**

Molecules coded as tensors of:
* atoms numbers shifted by 2 (e.g. hydrogen = 3).
  0 - reserved for padding, 1 - reserved for CLS token, 2 - extra reservation.
* neighbors count, including implicit hydrogens shifted by 2 (e.g. CO = CH3OH = [6, 4]).
  0 - reserved for padding, 1 - extra reservation, 2 - no-neighbors, 3 - one neighbor.
* topological distances' matrix shifted by 2 with upper limit.
  0 - reserved for padding, 1 - reserved for not-connected graph components coding, 2 - self-loop, 3 - connected atoms.

Reactions coded in similar way. Molecules atoms and neighbors matrices just stacked. Distance matrices stacked on diagonal.
Reactions include additional tensor with reaction role codes for each token.
0 - padding, 1 - reaction CLS, 2 - reactants, 3 - products.

    from chytorch.nn import MoleculeEncoder
    
    encoder = MoleculeEncoder()
    for b in dl:
        encoder(b)

**Combine molecules and labels:**

`chytorch.utils.data.chained_collate` - helper for combining different data parts. Useful for tricky input.

    from torch import stack
    from torch.utils.data import DataLoader, TensorDataset
    from chytorch.utils.data import chained_collate, collate_molecules, MoleculeDataset

    dl = DataLoader(TensorDataset(MoleculeDataset(molecules_list), properties_tensor),
        collate_fn=chained_collate(collate_molecules, stack))


**Scheduler:**

`chytorch.optim.lr_scheduler.WarmUpCosine` - Linear warmup followed with cosine-function for 0-pi range rescaled to lr_rate - decrease_coef * lr_rate interval.

**Voting NN with single hidden layer:**

`chytorch.nn.VotingClassifier`, `chytorch.nn.BinaryVotingClassifier` and `chytorch.nn.VotingRegressor` - speed optimized multiple heads for ensemble predictions.

**Helper Modules:**

`chytorch.nn.Slicer` - do tensor slicing. Useful for transformer's CLS token extraction in `torch.nn.Sequence`.

**Data Wrappers:**

In `chytorch.utils.data` module stored different data wrappers for simplifying ML workflows.
All wrappers have `torch.utils.data.Dataset` interface.

* `SizedList` - list wrapper with `size()` method. Useful with `torch.utils.data.TensorDataset`. 
* `SMILESDataset` - on-the-fly smiles to `chython.MoleculeContainer` or `chython.ReactionContainer` parser.
* `LMDBMapper` - LMDB KV storage to dataset mapper.
* `PostgresMapper` - Postgres DB table to dataset mapper.
* `SMILESTokenizerDataset` - on-the-fly generator of tokenized SMILES.
* `TensorUnpack`, `StructUnpack`, `PickleUnpack` - bytes to tensor/object unpackers


Publications
------------

[1](https://doi.org/10.1021/acs.jcim.2c00344) Bidirectional Graphormer for Reactivity Understanding: Neural Network Trained to Reaction Atom-to-Atom Mapping Task
