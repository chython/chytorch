Chytorch [kʌɪtɔːrtʃ]
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
Supported `chython.MoleculeContainer` and `chython.ReactionContainer` objects, and bytes-packed forms.

`chytorch.utils.data.collate_molecules` and `chytorch.utils.data.collate_reactions` - collate functions for `torch.utils.data.DataLoader`.

Example:

    from chython import SMILESRead
    from chytorch.utils.data import collate_molecules, MoleculeDataset
    from torch.utils.data import DataLoader

    data = []
    for r in SMILESRead('data.smi'):
        r.canonicalize()  # fix aromaticity and functional groups
        data.append(r)

    ds = MoleculeDataset(data)
    dl = DataLoader(ds, collate_fn=collate_molecules, batch_size=10)

**Forward call:**

Molecules coded as tensors of:
* atoms numbers shifted by 2 (e.g. hydrogen = 3).
  0 - reserved for padding, 1 - reserved for CLS token, 2 - for MLM task.
* neighbors count, including implicit hydrogens shifted by 2 (e.g. CO = CH3OH = [6, 4]).
  0 - reserved for padding, 1 - for MLM task, 2 - no-neighbors, 3 - one neighbor.
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

`chytorch.utils.data.chained_collate` - helper for combining different data parts. 

    from torch import stack
    from torch.utils.data import DataLoader, TensorDataset
    from chytorch.utils.data import chained_collate, collate_molecules, MoleculeDataset

    dl = DataLoader(TensorDataset(MoleculeDataset(molecules_list), properties_tensor),
        collate_fn=chained_collate(collate_molecules, torch.stack))


**Scheduler:**

`chytorch.optim.lr_scheduler.WarmUpCosine` - Linear warmup followed with cosine-function for 0-pi range rescaled to lr_rate - decrease_coef * lr_rate interval.

**Voting NN with single hidden layer:**

`chytorch.nn.VotingClassifier` and `chytorch.nn.VotingRegressor` - speed optimized multiple heads for ensemble predictions.


**Caching:**

`chytorch.utils.cache.SequencedFileCache`, `chytorch.utils.cache.SequencedDBCache`, `chytorch.utils.cache.SequencedDtypeCompressedCache`, `chytorch.utils.cache.CycleDataLoader` - helpers for caching slow dataset generators output.
