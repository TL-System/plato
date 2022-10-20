"""
The Lenet used by DLG paper
"""
import torch.nn as nn
from plato.config import Config


class Model(nn.Module):
    def __init__(self, num_classes=Config().trainer.num_classes):
        super().__init__()
        act = nn.Sigmoid
        if Config().data.datasource == 'EMNIST':
            in_channel = 1
            in_size = 588
        if Config().data.datasource.startswith('CIFAR'):
            in_channel = 3
            in_size = 768

        self.body = nn.Sequential(
            nn.Conv2d(in_channel, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_size, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        feature = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(feature)
        return out, feature

