import pandas as pd
import torch
import torch.nn as nn
from chython import smiles
from torch.utils.data import DataLoader

from chytorch.nn import MoleculeEncoder
from chytorch.utils import data
from chytorch.utils.data import chained_collate

BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCHS = 3


class Data:
    def __init__(self, path_to_csv):
        self.train_containers = None
        self.y_train = None
        self.test_containers = None
        self.y_test = None
        self.path_to_csv = path_to_csv

    def read_csv(self):
        df = pd.read_csv(self.path_to_csv)
        df = df[['std_smiles', 'activity', 'dataset']]
        mol_containers_train = [smiles(i) for i in df[df.dataset == 'train'].std_smiles]
        mol_containers_test = [smiles(i) for i in df[df.dataset == 'test'].std_smiles]
        y_train = df[df.dataset == 'train'].activity
        y_test = df[df.dataset == 'test'].activity
        self.train_containers = mol_containers_train
        self.y_train = torch.Tensor(y_train.to_list())
        self.test_containers = mol_containers_test
        self.y_test = torch.Tensor(y_test.to_list())

    def get_dataloader(self, data_type):
        if data_type == 'train':
            X_train = data.MoleculeDataset(self.train_containers, add_cls=True)
            dataset_loader = DataLoader(
                dataset=torch.utils.data.TensorDataset(X_train, self.y_train),
                collate_fn=chained_collate(data.collate_molecules, torch.stack),
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            return dataset_loader

        elif data_type == 'test':
            X_test = data.MoleculeDataset(self.test_containers, add_cls=True)
            dataset_loader = DataLoader(
                dataset=torch.utils.data.TensorDataset(X_test, self.y_train),
                collate_fn=chained_collate(data.collate_molecules, torch.stack),
                batch_size=BATCH_SIZE,
            )
            return dataset_loader


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.input_enсoder = MoleculeEncoder()
        self.linear = nn.Linear(in_features=1024,
                                out_features=1)  # https://stats.stackexchange.com/questions/207049/neural-network-for-binary-classification-use-1-or-2-output-neurons

    def forward(self, X):
        E = self.input_enсoder(X)
        X = self.linear(E[:,0])
        X = torch.relu(X)
        return X


def train(network, dataset_loader, loss_func, optimizer):
    size = len(dataset_loader.dataset)
    running_loss = []
    for batch, (X, y) in enumerate(dataset_loader):
        # compute prediction error
        predictions = network(X)
        loss = loss_func(predictions.squeeze(-1), y)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 1 == 0:
            loss_v, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>3f} [{current:>5d}/{size:>5d}]")
            running_loss.append(loss_v)  # count statistics


def test(network, dataset_loader, loss_func):
    size = len(dataset_loader.dataset)
    num_batches = len(dataset_loader)
    network.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataset_loader:
            predictions = network(X)
            test_loss += loss_func(predictions.squeeze(-1), y).item()
            correct += (predictions.argmax(1) == y).type(torch.float).sum().item()
    test_loss = test_loss/num_batches
    correct = correct/size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    dataset = Data(path_to_csv="")
    dataset.read_csv()
    train_dataloader = dataset.get_dataloader(data_type='train')
    test_dataloader = dataset.get_dataloader(data_type='test')
    network = Network()
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n------")
        train(network, train_dataloader, loss_func, optimizer)
        test(network, test_dataloader, loss_func)


