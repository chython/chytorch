import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from chython import smiles
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional, Union

from chytorch.nn import MoleculeEncoder, Slicer
from chytorch.utils.data import MoleculeDataset, chained_collate, collate_molecules

torch.manual_seed(1)

# check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class PandasData(pl.LightningDataModule):
    def __init__(
        self,
        csv: str,
        structure: str,
        property: str,
        dataset_type: str,
        batch_size: int = 32,
    ):
        super().__init__()
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.validation_x = None
        self.validation_y = None
        self.csv = csv
        self.structure = structure
        self.property = property
        self.dataset_type = dataset_type
        self.batch_size = batch_size

    def prepare_data(self):
        df = pd.read_csv(self.csv)
        df = df[[self.structure, self.property, self.dataset_type]]
        return df

    def setup(self, stage: Optional[str] = None):
        df = self.prepare_data()
        if stage == "fit" or stage is None:
            df_train = df[df.dataset == "train"]
            mols = [smiles(m) for m in df_train[self.structure]]
            [mol.kekule() for mol in mols]
            self.train_x = MoleculeDataset(mols)
            self.train_y = torch.Tensor(df_train[self.property].to_numpy())

        if stage == "validation" or stage is None:
            df_validation = df[df.dataset == "validation"]
            mols = [smiles(m) for m in df_validation[self.structure]]
            [mol.kekule() for mol in mols]
            self.validation_x = MoleculeDataset(mols)
            self.validation_y = torch.Tensor(df_validation[self.property].to_numpy())

        if stage == "test" or stage is None:
            df_test = df[df.dataset == "test"]
            mols = [smiles(m) for m in df_test[self.structure]]
            [mol.kekule() for mol in mols]
            self.test_x = MoleculeDataset(mols)
            self.test_y = torch.Tensor(df_test[self.property].to_numpy())

    def train_dataloader(self):
        return DataLoader(
            dataset=torch.utils.data.TensorDataset(self.train_x, self.train_y),
            collate_fn=chained_collate(collate_molecules, torch.stack),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=torch.utils.data.TensorDataset(self.train_x, self.train_y),
            collate_fn=chained_collate(collate_molecules, torch.stack),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=torch.utils.data.TensorDataset(self.test_x, self.test_y),
            collate_fn=chained_collate(collate_molecules, torch.stack),
            batch_size=self.batch_size,
        )


class Modeler:
    def __init__(
        self,
        loss_function,
        epochs: int,
        learning_rate: Union[float, int],
        csv: str,
        model_path: Optional[str] = None,
    ):
        self.network = None
        self.optimizer = None
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.csv = csv
        self.model_path = model_path

    def train_loop(self, dataset_loader: DataLoader):
        size = len(dataset_loader.dataset)
        for batch, (X, y) in enumerate(dataset_loader):
            # compute prediction and loss
            predictions = self.network(X)
            loss = self.loss_function(predictions.squeeze(-1), y)
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 10 == 0:
                loss_v, current = loss.item(), batch * len(X[0])
                print(f"loss: {loss_v:>3f} [{current:>5d}/{size:>5d}]")

    def validation_loop(self, dataset_loader: DataLoader):
        size = len(dataset_loader.dataset)
        num_batches = len(dataset_loader)
        self.network.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataset_loader:
                predictions = self.network(X)
                test_loss += self.loss_function(predictions.squeeze(-1), y).item()
                correct += (predictions.argmax(1) == y).type(torch.float).sum().item()
        test_loss = test_loss / num_batches
        correct = correct / size
        print(
            f"Validation Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    def save(self):
        torch.save(self.network.state_dict(), self.model_path)

    def fit(self):
        """
        Run model training
        """
        dataset = PandasData(
            csv=self.csv,
            structure="std_smiles",
            property="activity",
            dataset_type="dataset",
            batch_size=10,
        )
        dataset.setup()
        self.network = nn.Sequential(
            MoleculeEncoder(),
            Slicer(slice(None), 0),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid(),
        )
        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate)
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}\n------")
            self.train_loop(dataset.train_dataloader())
            self.validation_loop(dataset.test_dataloader())
        if self.model_path:
            self.save()


if __name__ == "__main__":
    modeler = Modeler(
        loss_function=nn.BCELoss(), epochs=3, learning_rate=2e-5, csv="", model_path=""
    )
    modeler.fit()
