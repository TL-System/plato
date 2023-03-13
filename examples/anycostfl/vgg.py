"""
In HeteroFL, the model needs to specifically designed to fit in the algorithm.
"""
import torch
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


class Scaler(nn.Module):
    "The scaler module for different rates of the models."

    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, feature):
        "Forward function."
        output = feature / self.rate if self.training else feature
        return output


class VGG(nn.Module):
    """
    VGG9 network.
    """

    def __init__(self, model_rate=1.0) -> None:
        super().__init__()

        self.scaler = Scaler(model_rate)
        channels = [
            int(channel * model_rate)
            for channel in [32, 64, 128, 128, 256, 256, 512, 512]
        ]

        self.conv1 = nn.Conv2d(3, channels[0], 3, 1, 1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, 1, 1)
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, 1, 1)
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, 1, 1)
        self.conv5 = nn.Conv2d(channels[3], channels[4], 3, 1, 1)
        self.conv6 = nn.Conv2d(channels[4], channels[5], 3, 1, 1)
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(16 * channels[5], channels[6])
        self.drop2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(channels[6], channels[7])
        self.drop3 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(channels[7], 10)
        self.apply(init_param)

    def forward(self, out):
        "Forward function."
        out = self.scaler(F.relu(self.conv1(out)))
        out = self.scaler(F.relu(self.conv2(out)))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.scaler(F.relu(self.conv3(out)))
        out = self.scaler(F.relu(self.conv4(out)))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.scaler(F.relu(self.conv5(out)))
        out = self.scaler(F.relu(self.conv6(out)))
        out = F.max_pool2d(out, kernel_size=2, stride=2)

        out = torch.flatten(out, 1)
        out = self.drop1(out)
        out = self.scaler(F.relu(self.fc1(out)))
        out = self.drop2(out)
        out = self.scaler(F.relu(self.fc2(out)))
        out = self.drop3(out)
        out = self.fc3(out)
        out = self.scaler(F.relu(out))
        return out


model = VGG()
# config = model.get_net(min)

import sys
import pickle
import ptflops

new = VGG(1)
size = sys.getsizeof(pickle.dumps(new.state_dict())) / 1024**2
macs, _ = ptflops.get_model_complexity_info(
    new,
    (3, 32, 32),
    as_strings=False,
    print_per_layer_stat=False,
    verbose=False,
)
macs /= 1024**2
print(size, macs)
