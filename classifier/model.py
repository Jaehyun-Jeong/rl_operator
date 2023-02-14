import torch
import torch.nn as nn

class Classifier(nn.Module):

    def __init__(self,
                 input_size,
                 output_size):
        self.input_size = input_size
        self.output_size = output_size

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.LeakyReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, 20),
            nn.LeakyReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)

        y = self.layers(x)
        # |y| = (batch_size, output_size)

        return y
