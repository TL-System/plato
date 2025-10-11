"""
The Multi-Layer Perception model for PyTorch.
The model follows the previous work to use tanh as activation
Reference: https://www.comp.nus.edu.sg/~reza/files/Shokri-SP2019.pdf
"""

import collections

import torch.nn as nn

from plato.config import Config


class Model(nn.Module):
    """The Multi-Layer Perception model.

    Arguments:
        num_classes (int): The number of classes. Default: 10.
    """

    def __init__(self, input_dim=600, num_classes=10):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 1024), nn.Tanh())

        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.Tanh())

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
        )

        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
        )

        self.fc5 = nn.Linear(128, num_classes)

        # Preparing named layers so that the model can be split and straddle
        # across the client and the server
        self.layers = []
        self.layerdict = collections.OrderedDict()
        self.layerdict["fc1"] = self.fc1
        self.layerdict["fc2"] = self.fc2
        self.layerdict["fc3"] = self.fc3
        self.layerdict["fc4"] = self.fc4
        self.layerdict["fc5"] = self.fc5

        self.layers.append("fc1")
        self.layers.append("fc2")
        self.layers.append("fc3")
        self.layers.append("fc4")
        self.layers.append("fc5")

    def forward(self, x):
        """Forward pass."""
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

    def forward_to(self, x, cut_layer):
        """Forward pass, but only to the layer specified by cut_layer."""
        layer_index = self.layers.index(cut_layer)

        for i in range(0, layer_index + 1):
            x = self.layerdict[self.layers[i]](x)

        return x

    def forward_from(self, x, cut_layer):
        """Forward pass, starting from the layer specified by cut_layer."""
        layer_index = self.layers.index(cut_layer)

        for i in range(layer_index + 1, len(self.layers)):
            x = self.layerdict[self.layers[i]](x)

        return x

    @staticmethod
    def get_model(*args):
        """Obtaining an instance of this model."""
        if hasattr(Config().trainer, "num_classes"):
            return Model(
                input_dim=Config().trainer.input_dim,
                num_classes=Config().trainer.num_classes,
            )
        return Model()
