""" The basic siamese network for mnist classification. """

import torch
import torch.nn as nn


class SiameseBase(nn.Module):

    def __init__(self):
        super(SiameseBase, self).__init__()

        # A simple two layer convolution followed by three fully connected layers should do

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3)

        self.lin1 = nn.Linear(144, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 10)

    def forward_once(self, x):

        # forwarding the input through the layers

        out = self.pool1(nn.functional.relu(self.conv1(x)))
        out = self.pool2(nn.functional.relu(self.conv2(out)))

        out = out.view(-1, 144)

        out = nn.functional.relu(self.lin1(out))
        out = nn.functional.relu(self.lin2(out))
        out = self.lin3(out)

        return out

    def forward(self, inputs):

        x, y = inputs
        # doing the forwarding twice so as to obtain the same functions as that of twin networks

        out1 = self.forward_once(x)
        out2 = self.forward_once(y)

        return (out1, out2)

    def evaluate(self, inputs):
        x, y = inputs
        # this can be used later for evalutation

        m = torch.tensor(1.0, dtype=torch.float32)

        if type(m) != type(x):
            x = torch.tensor(x, dtype=torch.float32, requires_grad=False)

        if type(m) != type(y):
            y = torch.tensor(y, dtype=torch.float32, requires_grad=False)

        x = x.view(-1, 1, 28, 28)
        y = y.view(-1, 1, 28, 28)

        with torch.no_grad():

            out1, out2 = self.forward((x, y))

            return nn.functional.pairwise_distance(out1, out2)