"""
In Fjord, the model needs to specifically designed to fit in the algorithm.
"""
import numpy as np
from torch import nn
import torch.nn.functional as F


def init_param(model):
    "Initialize the parameters of resnet."
    if isinstance(model, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        model.weight.data.fill_(1)
        model.bias.data.zero_()
    elif isinstance(model, nn.Linear):
        model.bias.data.zero_()
    return model


class Block(nn.Module):
    """
    ResNet block.
    """

    expansion = 1

    # pylint:disable=too-many-arguments
    def __init__(self, in_planes, planes, stride, track):
        super().__init__()
        bn1 = nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
        bn2 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        self.batchnorm1 = bn1
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.batchnorm2 = bn2
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, feature):
        "Forward function."
        out = F.relu(self.batchnorm1(feature))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else feature
        out = self.conv1(out)
        out = self.conv2(F.relu(self.batchnorm2(out)))
        out += shortcut
        return out


# pylint:disable=too-many-instance-attributes
class Bottleneck(nn.Module):
    """
    Bottleneck block
    """

    expansion = 4

    # pylint:disable=too-many-arguments
    def __init__(self, in_planes, planes, stride, track):
        super().__init__()
        bn1 = nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
        bn2 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        bn3 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        self.bn1 = bn1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = bn2
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = bn3
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, feature):
        """
        Forward function
        """
        out = F.relu(self.bn1(feature))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else feature
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    """
    The modified ResNet network.
    """

    # pylint:disable=too-many-arguments
    def __init__(self, data_shape, hidden_size, block, num_blocks, num_classes, track):
        super().__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(
            data_shape,
            hidden_size[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.layer1 = self._make_layer(
            block, hidden_size[0], num_blocks[0], stride=1, track=track
        )
        self.layer2 = self._make_layer(
            block, hidden_size[1], num_blocks[1], stride=2, track=track
        )
        self.layer3 = self._make_layer(
            block, hidden_size[2], num_blocks[2], stride=2, track=track
        )
        self.layer4 = self._make_layer(
            block, hidden_size[3], num_blocks[3], stride=2, track=track
        )
        bn4 = nn.BatchNorm2d(
            hidden_size[3] * block.expansion,
            momentum=None,
            track_running_stats=track,
        )
        self.bn4 = bn4
        self.linear = nn.Linear(hidden_size[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, track):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_index in strides:
            layers.append(block(self.in_planes, planes, stride_index, track))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, feature):
        """
        Forward function.
        """
        out = self.conv1(feature)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn4(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(model_rate=1, track=False):
    """ResNet18 model."""
    data_shape = 3
    classes_size = 10
    hidden_size = [int(np.ceil(model_rate * x)) for x in [64, 128, 256, 512]]
    model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], classes_size, track)
    model.apply(init_param)
    return model


def resnet34(model_rate=1, track=False):
    """ResNet34 model."""
    data_shape = 3
    classes_size = 10
    hidden_size = [int(np.ceil(model_rate * x)) for x in [64, 128, 256, 512]]
    model = ResNet(data_shape, hidden_size, Block, [3, 4, 6, 3], classes_size, track)
    model.apply(init_param)
    return model


def resnet50(model_rate=1, track=False):
    """ResNet50 model."""
    data_shape = 3
    classes_size = 10
    hidden_size = [int(np.ceil(model_rate * x)) for x in [64, 128, 256, 512]]
    model = ResNet(
        data_shape,
        hidden_size,
        Bottleneck,
        [3, 4, 6, 3],
        classes_size,
        track,
    )
    model.apply(init_param)
    return model


def resnet101(model_rate=1, track=False):
    """ResNet101 model."""
    data_shape = 3
    classes_size = 10
    hidden_size = [int(np.ceil(model_rate * x)) for x in [64, 128, 256, 512]]
    model = ResNet(
        data_shape,
        hidden_size,
        Bottleneck,
        [3, 4, 23, 3],
        classes_size,
        track,
    )
    model.apply(init_param)
    return model


def resnet152(model_rate=1, track=False):
    """ResNet152 model."""
    data_shape = 3
    classes_size = 10
    hidden_size = [int(np.ceil(model_rate * x)) for x in [64, 128, 256, 512]]
    model = ResNet(
        data_shape,
        hidden_size,
        Bottleneck,
        [3, 8, 36, 3],
        classes_size,
        track,
    )
    model.apply(init_param)
    return model
