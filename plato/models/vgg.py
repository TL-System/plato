"""
A VGG-style neural network model for image classification.
"""

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """A VGG-style neural network model for image classification."""

    class ConvModule(nn.Module):
        """A single convolutional module in a VGG network."""

        def __init__(self, in_filters, out_filters):
            super().__init__()
            self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_filters)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    def __init__(self, plan, outputs=10):
        super().__init__()

        layers = []
        filters = 3

        for spec in plan:
            if spec == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Model.ConvModule(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, outputs)

    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def is_valid_model_name(model_name):
        return (
            model_name.startswith("vgg")
            and len(model_name.split("_")) == 2
            and model_name.split("_")[1].isdigit()
            and int(model_name.split("_")[1]) in [11, 13, 16, 19]
        )

    @staticmethod
    def get(model_name, num_classes=10):
        """Returns a suitable VGG model corresponding to the provided name."""
        if not Model.is_valid_model_name(model_name):
            raise ValueError(f"Invalid VGG model name: {model_name}")

        outputs = num_classes

        num = int(model_name.split("_")[1])

        if num == 11:
            plan = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]
        elif num == 13:
            plan = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]
        elif num == 16:
            plan = [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
            ]
        elif num == 19:
            plan = [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                512,
            ]

        return Model(plan, outputs)
