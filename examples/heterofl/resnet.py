import numpy as np
from torch import nn
import torch.nn.functional as F


def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, rate, track):
        super(Block, self).__init__()
        n1 = nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
        n2 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        self.n1 = n1
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.n2 = n2
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scaler = Scaler(rate)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        out = F.relu(self.n1(self.scaler(x)))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(self.scaler(out))))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, rate, track):
        super(Bottleneck, self).__init__()
        n1 = nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
        n2 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        n3 = nn.BatchNorm2d(planes, momentum=None, track_running_stats=track)
        self.n1 = n1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.n2 = n2
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.n3 = n3
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.scaler = Scaler(rate)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        out = F.relu(self.n1(self.scaler(x)))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(self.scaler(out))))
        out = self.conv3(F.relu(self.n3(self.scaler(out))))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(
        self, data_shape, hidden_size, block, num_blocks, num_classes, rate, track
    ):
        super(ResNet, self).__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(
            data_shape[0],
            hidden_size[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.layer1 = self._make_layer(
            block, hidden_size[0], num_blocks[0], stride=1, rate=rate, track=track
        )
        self.layer2 = self._make_layer(
            block, hidden_size[1], num_blocks[1], stride=2, rate=rate, track=track
        )
        self.layer3 = self._make_layer(
            block, hidden_size[2], num_blocks[2], stride=2, rate=rate, track=track
        )
        self.layer4 = self._make_layer(
            block, hidden_size[3], num_blocks[3], stride=2, rate=rate, track=track
        )
        n4 = nn.BatchNorm2d(
            hidden_size[3] * block.expansion,
            momentum=None,
            track_running_stats=track,
        )

        self.linear = nn.Linear(hidden_size[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, rate, track):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rate, track))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        output = {}
        x = input["img"]
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.n4(self.scaler(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        output["score"] = out
        output["loss"] = F.cross_entropy(output["score"], input["label"])
        return output


def resnet18(model_rate=1, track=False):
    data_shape = 32
    classes_size = 10
    hidden_size = [int(np.ceil(model_rate * x)) for x in [64, 128, 256, 512]]
    scaler_rate = model_rate
    model = ResNet(
        data_shape, hidden_size, Block, [2, 2, 2, 2], classes_size, scaler_rate, track
    )
    model.apply(init_param)
    return model


def resnet34(model_rate=1, track=False):
    data_shape = 32
    classes_size = 10
    hidden_size = [int(np.ceil(model_rate * x)) for x in [64, 128, 256, 512]]
    scaler_rate = model_rate
    model = ResNet(
        data_shape, hidden_size, Block, [3, 4, 6, 3], classes_size, scaler_rate, track
    )
    model.apply(init_param)
    return model


def resnet50(model_rate=1, track=False):
    data_shape = 32
    classes_size = 10
    hidden_size = [int(np.ceil(model_rate * x)) for x in [64, 128, 256, 512]]
    scaler_rate = model_rate
    model = ResNet(
        data_shape,
        hidden_size,
        Bottleneck,
        [3, 4, 6, 3],
        classes_size,
        scaler_rate,
        track,
    )
    model.apply(init_param)
    return model


def resnet101(model_rate=1, track=False):
    data_shape = 32
    classes_size = 10
    hidden_size = [int(np.ceil(model_rate * x)) for x in [64, 128, 256, 512]]
    scaler_rate = model_rate
    model = ResNet(
        data_shape,
        hidden_size,
        Bottleneck,
        [3, 4, 23, 3],
        classes_size,
        scaler_rate,
        track,
    )
    model.apply(init_param)
    return model


def resnet152(model_rate=1, track=False):
    data_shape = 32
    classes_size = 10
    hidden_size = [int(np.ceil(model_rate * x)) for x in [64, 128, 256, 512]]
    scaler_rate = model_rate
    model = ResNet(
        data_shape,
        hidden_size,
        Bottleneck,
        [3, 8, 36, 3],
        classes_size,
        scaler_rate,
        track,
    )
    model.apply(init_param)
    return model
