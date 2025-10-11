# -*- coding: UTF-8 -*-

"""
MobileNetV3 From <Searching for MobileNetV3>, arXiv:1905.02244.
Ref: https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
     https://github.com/kuan-wang/pytorch-mobilenet-v3/blob/master/mobilenetv3.py

Modified specific for algorithm HeteroFL. sBN and scaler modules are implemented.
"""

from collections import OrderedDict

import torch.nn.functional as F
from torch import nn


def _ensure_divisible(number, divisor, min_value=None):
    """
    Ensure that 'number' can be 'divisor' divisible
    Reference from original tensorflow repo:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num


class Scaler(nn.Module):
    """
    Scaler module in HeteroFL
    """

    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, feature):
        "Forward function."
        output = feature / self.rate if self.training else feature
        return output


# pylint:disable=invalid-name
class H_sigmoid(nn.Module):
    """
    Hard sigmoid.
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, feature):
        "Forward function."
        return F.relu6(feature + 3, inplace=self.inplace) / 6


class H_swish(nn.Module):
    """
    Hard swish.
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, feature):
        "Forward function."
        return feature * F.relu6(feature + 3, inplace=self.inplace) / 6


class SEModule(nn.Module):
    """
    SE Module
    Ref: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """

    def __init__(self, in_channels_num, reduction_ratio=4):
        super().__init__()

        if in_channels_num % reduction_ratio != 0:
            raise ValueError(
                "in_channels_num must be divisible by reduction_ratio(default = 4)"
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels_num, in_channels_num // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels_num // reduction_ratio, in_channels_num, bias=False),
            H_sigmoid(),
        )

    def forward(self, x):
        "Forward function."
        batch_size, channel_num, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channel_num)
        y = self.fc(y).view(batch_size, channel_num, 1, 1)
        return x * y


class Bottleneck(nn.Module):
    """
    The basic unit of MobileNetV3.
    """

    # pylint:disable=too-many-arguments
    def __init__(
        self,
        in_channels_num,
        exp_size,
        out_channels_num,
        kernel_size,
        stride,
        use_SE,
        NL,
        BN_momentum,
        rate,
        tracking_stat,
        first,
    ):
        """
        use_SE: True or False -- use SE Module or not
        NL: nonlinearity, 'RE' or 'HS'
        """
        super().__init__()

        assert stride in [1, 2]
        NL = NL.upper()
        assert NL in ["RE", "HS"]

        use_HS = NL == "HS"

        # Whether to use residual structure or not
        self.use_residual = stride == 1 and in_channels_num == out_channels_num

        if first:
            # Without expansion, the first depthwise convolution is omitted
            self.conv1 = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(
                    in_channels=in_channels_num,
                    out_channels=exp_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    groups=in_channels_num,
                    bias=False,
                ),
                Scaler(rate),
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "lastBN",
                                nn.BatchNorm2d(
                                    num_features=exp_size,
                                    momentum=BN_momentum,
                                    track_running_stats=tracking_stat,
                                ),
                            )
                        ]
                    )
                ),
                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                # Linear Pointwise Convolution
                nn.Conv2d(
                    in_channels=exp_size,
                    out_channels=out_channels_num,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                Scaler(rate),
                # nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "lastBN",
                                nn.BatchNorm2d(
                                    num_features=out_channels_num,
                                    track_running_stats=tracking_stat,
                                ),
                            )
                        ]
                    )
                )
                if self.use_residual
                else nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "lastBN",
                                nn.BatchNorm2d(
                                    num_features=out_channels_num,
                                    momentum=BN_momentum,
                                    track_running_stats=tracking_stat,
                                ),
                            )
                        ]
                    )
                ),
            )
        else:
            # With expansion
            self.conv1 = nn.Sequential(
                # Pointwise Convolution for expansion
                nn.Conv2d(
                    in_channels=in_channels_num,
                    out_channels=exp_size,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                Scaler(rate),
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "lastBN",
                                nn.BatchNorm2d(
                                    num_features=exp_size,
                                    momentum=BN_momentum,
                                    track_running_stats=tracking_stat,
                                ),
                            )
                        ]
                    )
                ),
                H_swish() if use_HS else nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(
                    in_channels=exp_size,
                    out_channels=exp_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    groups=exp_size,
                    bias=False,
                ),
                Scaler(rate),
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "firstBN",
                                nn.BatchNorm2d(
                                    num_features=exp_size,
                                    momentum=BN_momentum,
                                    track_running_stats=tracking_stat,
                                ),
                            )
                        ]
                    )
                ),
                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True),
                # Linear Pointwise Convolution
                nn.Conv2d(
                    in_channels=exp_size,
                    out_channels=out_channels_num,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                Scaler(rate),
                # nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "lastBN",
                                nn.BatchNorm2d(
                                    num_features=out_channels_num,
                                    track_running_stats=tracking_stat,
                                ),
                            )
                        ]
                    )
                )
                if self.use_residual
                else nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "lastBN",
                                nn.BatchNorm2d(
                                    num_features=out_channels_num,
                                    momentum=BN_momentum,
                                    track_running_stats=tracking_stat,
                                ),
                            )
                        ]
                    )
                ),
            )

    def forward(self, x, expand=False):
        "Forward function"
        out1 = self.conv1(x)
        out = self.conv2(out1)
        if self.use_residual:
            out = out + x
        if expand:
            return out, out1
        return out


class MobileNetV3(nn.Module):
    """
    MobilenetV3 network.
    """

    # pylint:disable=too-many-arguments
    # pylint:disable=too-many-locals
    def __init__(
        self,
        mode="small",
        classes_num=10,
        input_size=32,
        width_multiplier=1.0,
        dropout=0.2,
        BN_momentum=0.1,
        zero_gamma=False,
        model_rate=1.0,
        track=True,
    ):
        """
        configs: setting of the model
        mode: type of the model, 'large' or 'small'
        """
        super().__init__()

        mode = mode.lower()
        assert mode in ["large", "small"]
        self.rate = model_rate
        self.tracking_stat = track

        s = 2
        if input_size in [32, 56]:
            # using cifar-10, cifar-100 or Tiny-ImageNet
            s = 1

        # setting of the model
        if mode == "large":
            # Configuration of a MobileNetV3-Large Model
            configs = [
                # kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, False, "RE", 1, True],
                [3, 64, 24, False, "RE", s, False],
                [3, 72, 24, False, "RE", 1, False],
                [5, 72, 40, True, "RE", 2, False],
                [5, 120, 40, True, "RE", 1, False],
                [5, 120, 40, True, "RE", 1, False],
                [3, 240, 80, False, "HS", 2, False],
                [3, 200, 80, False, "HS", 1, False],
                [3, 184, 80, False, "HS", 1, False],
                [3, 184, 80, False, "HS", 1, False],
                [3, 480, 112, True, "HS", 1, False],
                [3, 672, 112, True, "HS", 1, False],
                [5, 672, 160, True, "HS", 2, False],
                [5, 960, 160, True, "HS", 1, False],
                [5, 960, 160, True, "HS", 1, False],
            ]
        elif mode == "small":
            # Configuration of a MobileNetV3-Small Model
            configs = [
                # kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, True, "RE", s, True],
                [3, 72, 24, False, "RE", 2, False],
                [3, 88, 24, False, "RE", 1, False],
                [5, 96, 40, True, "HS", 2, False],
                [5, 240, 40, True, "HS", 1, False],
                [5, 240, 40, True, "HS", 1, False],
                [5, 120, 48, True, "HS", 1, False],
                [5, 144, 48, True, "HS", 1, False],
                [5, 288, 96, True, "HS", 2, False],
                [5, 576, 96, True, "HS", 1, False],
                [5, 576, 96, True, "HS", 1, False],
            ]
        for index, config in enumerate(configs):
            configs[index][1] = max(int(self.rate * config[1]), 1)
            configs[index][2] = max(int(self.rate * config[2]), 1)

        first_channels_num = int(16 * self.rate)

        last_channels_num = int(1280 if mode == "large" else 1024 * self.rate)

        divisor = 8

        # feature extraction part
        # input layer
        input_channels_num = _ensure_divisible(
            first_channels_num * width_multiplier, divisor
        )
        last_channels_num = (
            _ensure_divisible(last_channels_num * width_multiplier, divisor)
            if width_multiplier > 1
            else last_channels_num
        )
        feature_extraction_layers = []
        first_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=input_channels_num,
                kernel_size=3,
                stride=s,
                padding=1,
                bias=False,
            ),
            Scaler(self.rate),
            nn.BatchNorm2d(
                num_features=input_channels_num,
                momentum=BN_momentum,
                track_running_stats=self.tracking_stat,
            ),
            H_swish(),
        )
        feature_extraction_layers.append(first_layer)
        # Overlay of multiple bottleneck structures
        for (
            kernel_size,
            exp_size,
            out_channels_num,
            use_SE,
            NL,
            stride,
            first,
        ) in configs:
            output_channels_num = _ensure_divisible(
                out_channels_num * width_multiplier, divisor
            )
            exp_size = _ensure_divisible(exp_size * width_multiplier, divisor)
            feature_extraction_layers.append(
                Bottleneck(
                    input_channels_num,
                    exp_size,
                    output_channels_num,
                    kernel_size,
                    stride,
                    use_SE,
                    NL,
                    BN_momentum,
                    self.rate,
                    self.tracking_stat,
                    first,
                )
            )
            input_channels_num = output_channels_num

        # the last stage
        last_stage_channels_num = _ensure_divisible(
            exp_size * width_multiplier, divisor
        )
        last_stage_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels_num,
                out_channels=last_stage_channels_num,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            Scaler(self.rate),
            nn.BatchNorm2d(
                num_features=last_stage_channels_num,
                momentum=BN_momentum,
                track_running_stats=self.tracking_stat,
            ),
            H_swish(),
        )
        feature_extraction_layers.append(last_stage_layer1)

        self.featureList = nn.ModuleList(feature_extraction_layers)
        last_stage = []
        last_stage.append(nn.AdaptiveAvgPool2d(1))
        last_stage.append(
            nn.Conv2d(
                in_channels=last_stage_channels_num,
                out_channels=last_channels_num,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        last_stage.append(H_swish())

        self.last_stage_layers = nn.Sequential(*last_stage)
        # Classification part

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(last_channels_num, classes_num)
        )
        # Initialize the weights
        self._initialize_weights(zero_gamma)

    def forward(self, x):
        "Forward function."
        for i in range(9):
            x = self.featureList[i](x)
        x = self.featureList[9](x)
        for i in range(10, len(self.featureList)):
            x = self.featureList[i](x)
        x = self.last_stage_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, zero_gamma):
        """
        Initialize the weights
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if zero_gamma:
            for m in self.modules():
                if hasattr(m, "lastBN"):
                    nn.init.constant_(m.lastBN.weight, 0.0)
