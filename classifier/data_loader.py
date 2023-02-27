import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class OperationDataset(Dataset):

    # It's because smallest number is -9
    INDEXING_GAP = 9

    def __init__(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            ):

        self.data = x
        self.labels = y
        self.indexing()

        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        return x, y

    def indexing(self):
        for i in range(self.labels.size(0)):
            self.labels[i] = self.labels[i] + self.INDEXING_GAP


def get_loaders(config):

    df = pd.read_csv("../data.csv")

    x = torch.Tensor(df.iloc[:, :-1].values)
    y = torch.tensor(df.iloc[:, -1].values)

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    indices = torch.randperm(x.size(0))
    train_x, valid_x = torch.index_select(
            x,
            dim=0,
            index=indices
        ).split([train_cnt, valid_cnt], dim=0)
    train_y, valid_y = torch.index_select(
            y,
            dim=0,
            index=indices
        ).split([train_cnt, valid_cnt], dim=0)

    train_loader = DataLoader(
            dataset=OperationDataset(train_x, train_y),
            batch_size=config.batch_size,
            shuffle=True)
    valid_loader = DataLoader(
            dataset=OperationDataset(valid_x, valid_y),
            batch_size=config.batch_size,
            shuffle=True)

    # Load Test

    df = pd.read_csv("../test.csv")

    test_x = torch.tensor(df.iloc[:, :-1].values)
    test_y = torch.tensor(df.iloc[:, -1].values)
    test_loader = DataLoader(
            dataset=OperationDataset(test_x, test_y),
            batch_size=config.batch_size,
            shuffle=False)

    return train_loader, valid_loader, test_loader
