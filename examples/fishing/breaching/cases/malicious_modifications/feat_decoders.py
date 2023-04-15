"""Models that invert feature representations for use in VAE-style minimal representation attacks."""
import torch
from ..models.utils import get_layer_functions


class BasicDecodingBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        upsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        conv=torch.nn.Conv2d,
        nonlin=torch.nn.ReLU,
        norm_layer=torch.nn.BatchNorm2d,
        bias=False,
    ):
        super().__init__()
        self.interpolate = torch.nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)
        self.conv1 = conv(planes, planes, kernel_size=3, stride=1, padding=1, groups=1, bias=bias, dilation=1)
        self.bn1 = norm_layer(planes)
        self.nonlin = nonlin()
        self.conv2 = conv(planes, inplanes, kernel_size=3, stride=1, padding=1, groups=1, bias=bias, dilation=1)
        self.bn2 = norm_layer(inplanes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(self.interpolate(x))
        out = self.bn1(out)
        out = self.nonlin(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.nonlin(out)

        return out


class BottleneckDecoding(torch.nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        upsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        conv=torch.nn.Conv2d,
        nonlin=torch.nn.ReLU,
        norm_layer=torch.nn.BatchNorm2d,
        bias=False,
    ):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv(planes * self.expansion, width, kernel_size=1, stride=1, bias=bias)
        self.bn1 = norm_layer(width)

        self.interpolate = torch.nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)
        self.conv2 = conv(
            width, width, kernel_size=3, stride=1, padding=dilation, groups=groups, bias=bias, dilation=dilation
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv(width, inplanes, kernel_size=1, stride=1, bias=bias)
        self.bn3 = norm_layer(inplanes)
        self.nonlin = nonlin()
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlin(out)

        out = self.conv2(self.interpolate(out))
        out = self.bn2(out)
        out = self.nonlin(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.nonlin(out)

        return out


class ResNetDecoder(torch.nn.Module):
    def __init__(
        self,
        block,
        layers,
        channels,
        classes,
        zero_init_residual=False,
        strides=[1, 2, 2, 2],
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=[False, False, False, False],
        norm="BatchNorm2d",
        nonlin="ReLU",
        stem="CIFAR",
        upsample="B",
        convolution_type="Standard",
    ):
        super().__init__()

        self._conv_layer, self._norm_layer, self._nonlin_layer = get_layer_functions(convolution_type, norm, nonlin)
        self.use_bias = False
        self.inplanes = width_per_group if isinstance(block, BasicDecodingBlock) else 64
        self.feature_width = self.inplanes * 2 ** len(layers) // 2
        self.dilation = 1
        if len(replace_stride_with_dilation) != 4:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 4-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group if isinstance(block, BottleneckDecoding) else 64

        scale = 4 if stem == "CIFAR" else 7
        self.interpolate = torch.nn.Upsample(scale_factor=scale, mode="nearest", align_corners=None)

        layer_list = []
        self.target_width = int(self.feature_width)
        for idx, layer in reversed(list(enumerate(layers))):
            print(block, self.target_width, self.feature_width, layer, strides[idx])
            layer_list.append(
                self._make_layer(
                    block,
                    self.feature_width,
                    layer,
                    stride=strides[idx],
                    dilate=replace_stride_with_dilation[idx],
                    upsample=upsample,
                )
            )
            self.feature_width = self.target_width
        self.layers = torch.nn.Sequential(*layer_list)

        if stem == "CIFAR":
            conv1 = self._conv_layer(
                self.inplanes, channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1
            )
            bn1 = self._norm_layer(channels)
            nonlin = torch.nn.Tanh()
            self.stem = torch.nn.Sequential(conv1, bn1, nonlin)
        elif stem == "standard":
            interpolate0 = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            conv1 = self._conv_layer(self.inplanes, channels, kernel_size=7, stride=1, padding=3, bias=self.use_bias)
            bn1 = self._norm_layer(channels)
            interpolate1 = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            nonlin = torch.nn.Tanh()
            self.stem = torch.nn.Sequential(interpolate0, conv1, bn1, interpolate, nonlin)
        else:
            raise ValueError(f"Invalid stem designation {stem}.")

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckDecoding):
                    if hasattr(m.bn3, "weight"):
                        torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicDecodingBlock):
                    if hasattr(m.bn2, "weight"):
                        torch.nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, upsample="B"):
        conv_layer = self._conv_layer
        norm_layer = self._norm_layer
        nonlin_layer = self._nonlin_layer
        upsample_op = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        planes = int(planes) // block.expansion
        if stride != 1:
            self.target_width //= 2
            if upsample == "A":
                upsample_op = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False),
                    conv_layer(
                        planes * block.expansion, self.target_width, kernel_size=1, stride=1, bias=self.use_bias
                    ),
                )
            elif upsample == "B":
                upsample_op = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False),
                    conv_layer(
                        planes * block.expansion, self.target_width, kernel_size=1, stride=1, bias=self.use_bias
                    ),
                    norm_layer(self.target_width),
                )
            elif upsample == "C":
                upsample_op = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False),
                    conv_layer(
                        planes * block.expansion, self.target_width, kernel_size=1, stride=1, bias=self.use_bias
                    ),
                    norm_layer(self.target_width),
                )
            elif upsample == "preact-B":
                upsample_op = torch.nn.Sequential(
                    nonlin_layer(),
                    torch.nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False),
                    conv_layer(
                        planes * block.expansion, self.target_width, kernel_size=1, stride=1, bias=self.use_bias
                    ),
                )
            elif upsample == "preact-C":
                upsample_op = torch.nn.Sequential(
                    nonlin_layer(),
                    torch.nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False),
                    conv_layer(
                        planes * block.expansion, self.target_width, kernel_size=1, stride=1, bias=self.use_bias
                    ),
                )
            else:
                raise ValueError("Invalid upsample block specification.")

        layers = []

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.feature_width,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    nonlin=nonlin_layer,
                    bias=self.use_bias,
                )
            )

        layers.append(
            block(
                self.target_width,
                planes,
                stride,
                upsample_op,
                self.groups,
                self.base_width,
                previous_dilation,
                conv=conv_layer,
                nonlin=nonlin_layer,
                norm_layer=norm_layer,
                bias=self.use_bias,
            )
        )

        return torch.nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = x[:, :, None, None]  # Introduce spatial dimensions
        x = self.interpolate(x)
        x = self.layers(x)
        x = self.stem(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def generate_decoder(original_model):
    """For now only resnets with a BasicBlock are possible and only CIFAR10 :<

    In the future this function would ideally generate the decoder only up to the input resolution."""
    layers = [len(layer) for layer in original_model.layers]
    model = ResNetDecoder(
        BasicDecodingBlock,
        layers,
        3,
        10,
        stem="CIFAR",
        convolution_type="Standard",
        nonlin="ReLU",
        norm="BatchNorm2d",
        upsample="B",
        width_per_group=64,
        zero_init_residual=False,
    )
    return model
