"""
The ResNet model used in Geiping's implementation.

Reference:
Geiping et al., "Inverting Gradients - How easy is it to break privacy in federated learning?,"
in the Proceedings of NeurIPS 2020.
https://github.com/JonasGeiping/invertinggradients
"""

from __future__ import annotations

import random
from typing import Callable, Sequence, cast

import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

from plato.config import Config

BlockType = type[BasicBlock] | type[Bottleneck]


def set_random_seed(seed: int = 233) -> None:
    """233 = 144 + 89 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)


class Model(nn.Module):
    """ResNet generalization for CIFAR variants."""

    def __init__(
        self,
        block: BlockType,
        layers: Sequence[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        base_width: int = 64,
        replace_stride_with_dilation: Sequence[bool] | None = None,
        norm_layer: type[nn.Module] | None = None,
        strides: Sequence[int] = (1, 2, 2, 2),
        pool: str = "avg",
    ):
        """Initialize as usual. Layers and strides are scriptable."""
        super().__init__()

        norm_layer_cls = cast(Callable[[int], nn.Module], norm_layer or nn.BatchNorm2d)

        if replace_stride_with_dilation is None:
            replace = [False] * len(layers)
        else:
            replace = list(replace_stride_with_dilation)

        if len(replace) < len(layers):
            raise ValueError(
                "replace_stride_with_dilation must match number of layers."
            )

        stride_list = list(strides)
        if len(stride_list) < len(layers):
            raise ValueError("strides must match number of layers.")

        block_base_width = (
            64  # BasicBlock requires base_width=64; this matches torchvision.
        )

        self.conv1 = nn.Conv2d(
            3, base_width, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer_cls(base_width)
        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList()
        width = base_width
        inplanes = base_width
        dilation = 1

        for idx, block_count in enumerate(layers):
            layer_module, inplanes, dilation = self._make_layer(
                block=block,
                inplanes=inplanes,
                planes=width,
                blocks=block_count,
                stride=stride_list[idx],
                dilate=replace[idx],
                groups=groups,
                base_width=block_base_width,
                dilation=dilation,
                norm_layer=norm_layer_cls,
            )
            self.layers.append(layer_module)
            width *= 2

        self.pool = (
            nn.AdaptiveAvgPool2d((1, 1))
            if pool == "avg"
            else nn.AdaptiveMaxPool2d((1, 1))
        )
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck) and isinstance(
                    module.bn3.weight, torch.Tensor
                ):
                    nn.init.constant_(module.bn3.weight, 0.0)
                elif isinstance(module, BasicBlock) and isinstance(
                    module.bn2.weight, torch.Tensor
                ):
                    nn.init.constant_(module.bn2.weight, 0.0)

    def _make_layer(
        self,
        *,
        block: BlockType,
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int,
        dilate: bool,
        groups: int,
        base_width: int,
        dilation: int,
        norm_layer: Callable[[int], nn.Module],
    ) -> tuple[nn.Sequential, int, int]:
        """Build a residual layer stack, mirroring torchvision's implementation."""
        downsample: nn.Module | None = None
        previous_dilation = dilation
        stride_to_use = stride
        updated_dilation = dilation

        if dilate:
            updated_dilation *= stride
            stride_to_use = 1

        if stride_to_use != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride_to_use),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                inplanes,
                planes,
                stride_to_use,
                downsample,
                groups,
                base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        current_inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    current_inplanes,
                    planes,
                    groups=groups,
                    base_width=base_width,
                    dilation=updated_dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers), current_inplanes, updated_dilation

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Model forwarding."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Standard forward pass matching torchvision's API."""
        return self._forward_impl(x)

    def forward_feature(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Model forwarding returning intermediate feature."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        feature = x.clone()
        x = self.fc(x)

        return x, feature

    @staticmethod
    def is_valid_model_type(model_type: str | None) -> bool:
        return (
            isinstance(model_type, str)
            and model_type.startswith("resnet_")
            and len(model_type.split("_")) == 2
            and int(model_type.split("_")[1]) in [18, 34, 50, 101, 152, 32]
        )

    @staticmethod
    def resnet18() -> "Model":
        return Model(
            BasicBlock,
            [2, 2, 2, 2],
            Config().parameters.model.num_classes,
            base_width=64,
        )

    @staticmethod
    def resnet32() -> "Model":
        return Model(
            BasicBlock,
            [5, 5, 5],
            Config().parameters.model.num_classes,
            base_width=16 * 10,
        )

    @staticmethod
    def resnet34() -> "Model":
        return Model(
            BasicBlock,
            [3, 4, 6, 3],
            Config().parameters.model.num_classes,
            base_width=64,
        )

    @staticmethod
    def resnet50() -> "Model":
        return Model(
            Bottleneck,
            [3, 4, 6, 3],
            Config().parameters.model.num_classes,
            base_width=64,
        )

    @staticmethod
    def resnet101() -> "Model":
        return Model(
            Bottleneck,
            [3, 4, 23, 3],
            Config().parameters.model.num_classes,
            base_width=64,
        )

    @staticmethod
    def resnet152() -> "Model":
        return Model(
            Bottleneck,
            [3, 8, 36, 3],
            Config().parameters.model.num_classes,
            base_width=64,
        )


def get(model_name: str | None = None) -> Callable[[], Model] | None:
    """Returns a suitable ResNet model according to its type."""
    set_random_seed(1)

    if not Model.is_valid_model_type(model_name):
        raise ValueError(f"Invalid Resnet model name: {model_name}")

    assert model_name is not None
    resnet_type = int(model_name.split("_")[1])

    if resnet_type == 18:
        return Model.resnet18
    if resnet_type == 32:
        return Model.resnet32
    if resnet_type == 34:
        return Model.resnet34
    if resnet_type == 50:
        return Model.resnet50
    if resnet_type == 101:
        return Model.resnet101
    if resnet_type == 152:
        return Model.resnet152
    return None
