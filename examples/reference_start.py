import pandas as pd
import torch
import torch.nn as nn
from chython import smiles
from torch.utils.data import DataLoader

from chytorch.nn import MoleculeEncoder
from chytorch.utils import data
from chytorch.utils.data import chained_collate

BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 3
torch.manual_seed(1)

# check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class Data:
    def __init__(self, path_to_csv):
        self.train_containers = None
        self.y_train = None
        self.test_containers = None
        self.y_test = None
        self.path_to_csv = path_to_csv

    def read_csv(self):
        print("Read csv file")
        df = pd.read_csv(self.path_to_csv)
        df = df[["std_smiles", "activity", "dataset"]]
        mol_containers_train = [
            smiles(i) for i in df[df.dataset == "train"].std_smiles[:10000]
        ]
        mol_containers_test = [
            smiles(i) for i in df[df.dataset == "test"].std_smiles[:100]
        ]
        y_train = df[df.dataset == "train"].activity[:10000]
        y_test = df[df.dataset == "test"].activity[:100]
        self.train_containers = mol_containers_train
        self.y_train = torch.Tensor(y_train.to_list())
        self.test_containers = mol_containers_test
        self.y_test = torch.Tensor(y_test.to_list())

    def get_dataloader(self, data_type):
        print("Prepare dataloader")
        if data_type == "train":
            X_train = data.MoleculeDataset(self.train_containers, add_cls=True)
            dataset_loader = DataLoader(
                dataset=torch.utils.data.TensorDataset(X_train, self.y_train),
                collate_fn=chained_collate(data.collate_molecules, torch.stack),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            return dataset_loader

        elif data_type == "test":
            X_test = data.MoleculeDataset(self.test_containers, add_cls=True)
            dataset_loader = DataLoader(
                dataset=torch.utils.data.TensorDataset(X_test, self.y_test),
                collate_fn=chained_collate(data.collate_molecules, torch.stack),
                batch_size=BATCH_SIZE,
            )
            return dataset_loader


class Network(torch.nn.Module):
    def __init__(self):
        """
        Define model
        """
        super(Network, self).__init__()
        self.input_encoder = MoleculeEncoder()
        self.linear = nn.Sequential(
            nn.Linear(in_features=1024, out_features=516),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=516, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, X):
        E = self.input_encoder(X)
        X = self.linear(E[:, 0])
        return X


def train_loop(network, dataset_loader, loss_func, optimizer):
    size = len(dataset_loader.dataset)
    for batch, (X, y) in enumerate(dataset_loader, 1):
        # compute prediction and loss
        predictions = network(X)
        loss = loss_func(predictions.squeeze(-1), y)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss_v, current = loss.item(), batch * len(X[0])
            print(f"loss: {loss_v:>3f} [{current:>5d}/{size:>5d}]")


def test_loop(network, dataset_loader, loss_func):
    size = len(dataset_loader.dataset)
    num_batches = len(dataset_loader)
    network.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataset_loader:
            predictions = network(X)
            test_loss += loss_func(predictions.squeeze(-1), y).item()
            correct += (predictions.argmax(1) == y).type(torch.float).sum().item()
    test_loss = test_loss / num_batches
    correct = correct / size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def train_model(path_to_data):
    """
    Run model training
    """
    dataset = Data(path_to_csv=path_to_data)
    dataset.read_csv()
    train_dataloader = dataset.get_dataloader(data_type="train")
    test_dataloader = dataset.get_dataloader(data_type="test")
    loss_func = nn.BCELoss()
    network = Network()
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n------")
        train_loop(network, train_dataloader, loss_func, optimizer)
        test_loop(network, test_dataloader, loss_func)


if __name__ == "__main__":
    train_model("PATH_TO_CSV")
