"""
The LeNet model used in the Zhu's implementation.

Reference:
Zhu et al., "Deep Leakage from Gradients," in the Proceedings of NeurIPS 2019.
https://github.com/mit-han-lab/dlg
"""

import torch.nn as nn

from plato.config import Config


class Model(nn.Module):
    """The LeNet model."""

    def __init__(self, num_classes=Config().parameters.model.num_classes):
        super().__init__()
        act = nn.Sigmoid
        if Config().data.datasource in ["MNIST", "EMNIST"]:
            in_channel = 1
            in_size = 588
        if Config().data.datasource.startswith("CIFAR"):
            in_channel = 3
            in_size = 768

        self.body = nn.Sequential(
            nn.Conv2d(in_channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(nn.Linear(in_size, num_classes))

    def forward(self, x):
        """Model forwarding."""
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def forward_feature(self, x):
        """Model forwarding returning intermediate feature."""
        out = self.body(x)
        feature = out.view(out.size(0), -1)
        out = self.fc(feature)

        return out, feature
