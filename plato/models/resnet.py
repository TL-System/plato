"""
The ResNet model (for the CIFAR-10 dataset only).

Reference:

https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""
import torch
import torch.nn as nn
import torchvision


class Model(torchvision.models.ResNet):
    """ResNet generalization for CIFAR thingies."""

    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        base_width=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        strides=[1, 2, 2, 2],
        pool="avg",
    ):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 4-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layers = torch.nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(
                self._make_layer(
                    block,
                    width,
                    layer,
                    stride=strides[idx],
                    dilate=replace_stride_with_dilation[idx],
                )
            )
            width *= 2

        self.pool = (
            nn.AdaptiveAvgPool2d((1, 1))
            if pool == "avg"
            else nn.AdaptiveMaxPool2d((1, 1))
        )
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        # TODO: move initialization?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, torchvision.models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision.models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

class Model(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cut_layer=None):
        super().__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # CIFAR10
        if num_classes == 10:
            self.linear = nn.Linear(512 * block.expansion, num_classes)
        # ImageNet
        else:
            self.linear = nn.Linear(41472 * block.expansion, num_classes)

        # Preparing named layers so that the model can be split and straddle
        # across the client and the server
        self.layers = []
        self.layerdict = collections.OrderedDict()
        self.layerdict["conv1"] = self.conv1
        self.layerdict["bn1"] = self.bn1
        self.layerdict["relu"] = F.relu
        self.layerdict["layer1"] = self.layer1
        self.layerdict["layer2"] = self.layer2
        self.layerdict["layer3"] = self.layer3
        self.layerdict["layer4"] = self.layer4
        self.layers.append("conv1")
        self.layers.append("bn1")
        self.layers.append("relu")
        self.layers.append("layer1")
        self.layers.append("layer2")
        self.layers.append("layer3")
        self.layers.append("layer4")
        self.cut_layer = cut_layer

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feature = out
        out = self.linear(out)
        return out, feature

    def forward_to(self, x):
        """Forward pass, but only to the layer specified by cut_layer."""
        layer_index = self.layers.index(self.cut_layer)

        for i in range(0, layer_index + 1):
            x = self.layerdict[self.layers[i]](x)
        return x

    def forward_from(self, x):
        """Forward pass, starting from the layer specified by cut_layer."""
        layer_index = self.layers.index(self.cut_layer)
        for i in range(layer_index + 1, len(self.layers)):
            x = self.layerdict[self.layers[i]](x)

        out = F.avg_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @staticmethod
    def is_valid_model_type(model_type):
        return (
            model_type.startswith("resnet_")
            and len(model_type.split("_")) == 2
            and int(model_type.split("_")[1]) in [18, 34, 50, 101, 152, 32]
        )

    @staticmethod
    def get(model_name=None, num_classes=None, cut_layer=None, **kwargs):
        """Returns a suitable ResNet model according to its type."""
        if not Model.is_valid_model_type(model_name):
            raise ValueError(f"Invalid Resnet model name: {model_name}")

        resnet_type = int(model_name.split("_")[1])

        if num_classes is None:
            num_classes = 10

        if resnet_type == 18:
            return Model(BasicBlock, [2, 2, 2, 2], num_classes, cut_layer)
        elif resnet_type == 34:
            return Model(BasicBlock, [3, 4, 6, 3], num_classes, cut_layer)
        elif resnet_type == 50:
            return Model(Bottleneck, [3, 4, 6, 3], num_classes, cut_layer)
        elif resnet_type == 101:
            return Model(Bottleneck, [3, 4, 23, 3], num_classes, cut_layer)
        elif resnet_type == 152:
            return Model(Bottleneck, [3, 8, 36, 3], num_classes, cut_layer)

        return None
