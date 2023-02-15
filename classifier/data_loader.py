import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class OperationDataset(Dataset):

    def __init__(
            self,
            path: str,
            ):

        df = pd.read_csv(path)

        self.data = df.iloc[:, :-1].values
        self.labels = df.iloc[:, -1].values

        super().__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        return x, y


def get_loaders(config):
    x, y = load_mnist(is_train = True, flatten = False)

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    indices = torch.randperm(x.size(0))
    train_x, valid_x = torch.index_select(
            x,
            dim = 0,
            index = indices
        ).split([train_cnt, valid_cnt], dim = 0)
    train_y, valid_y = torch.index_select(
            y,
            dim = 0,
            index = indices
        ).split([train_cnt, valid_cnt], dim = 0)

    train_loader = DataLoader(
            dataset = MnistDataset(train_x, train_y, flatten = True),
            batch_size = config.batch_size,
            shuffle = True)
    valid_loader = DataLoader(
            dataset = MnistDataset(valid_x, valid_y, flatten = True),
            batch_size = config.batch_size,
            shuffle = True)

    test_x, test_y = load_mnist(is_train = False, flatten = False)
    test_loader = DataLoader(
            dataset = MnistDataset(test_x, test_y, flatten = True),
            batch_size = config.batch_size,
            shuffle = False)

    return train_loader, valid_loader, test_loader
