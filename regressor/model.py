import torch.nn as nn


class Regressor(nn.Module):

    def __init__(self,
                 input_size):
        self.input_size = input_size

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(5),
            nn.Linear(5, 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4, 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)

        y = self.layers(x)
        # |y| = (batch_size, output_size)

        return y.squeeze()
