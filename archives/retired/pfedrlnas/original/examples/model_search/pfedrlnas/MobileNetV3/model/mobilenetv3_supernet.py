"""
The replicated search space of MobileNetV3.

Reference:
Searching for MobileNetV3.
https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf.
https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html.
"""

import collections
import sys

from torch import nn

from plato.config import Config

from .config import get_config

sys.path.append("../")  # Add parent directory to path to access VIT module
from VIT.nasvit_wrapper.attentive_nas_dynamic_model import (
    AttentiveNasDynamicModel,
)
from VIT.nasvit_wrapper.dynamic_layers import (
    DynamicConvBnActLayer,
    DynamicLinearLayer,
    DynamicMBConvLayer,
    DynamicShortcutLayer,
)
from VIT.nasvit_wrapper.NASViT.models.modules.nn_utils import int2list
from VIT.nasvit_wrapper.NASViT.models.modules.static_layers import (
    MobileInvertedResidualBlock,
)


class NasDynamicModel(AttentiveNasDynamicModel):
    """The supernet for Mobilenetv3."""

    # pylint:disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods
    def __init__(self, supernet=None, n_classes=-1, bn_param=(0.0, 1e-5)):
        if supernet is None:
            supernet = get_config().supernet_config
        super().__init__()
        self.initialization(supernet, n_classes, bn_param)

    def initialization(self, supernet=None, n_classes=-1, bn_param=(0.0, 1e-5)):
        """
        Initilization function.
        """
        # pylint:disable=too-many-locals
        # pylint:disable=too-many-statements
        if supernet is None:
            supernet = get_config().supernet_config
        if n_classes == -1:
            n_classes = Config().parameters.model.num_classes
        bn_momentum = Config().parameters.model.bn_momentum
        bn_eps = Config().parameters.model.bn_eps
        bn_param = (bn_momentum, bn_eps)

        self.supernet = supernet
        self.n_classes = n_classes
        self.init_cfg_candidates()

        # first conv layer, including conv, bn, act
        out_channel_list, act_func, stride = (
            self.supernet.first_conv.c,
            self.supernet.first_conv.act_func,
            self.supernet.first_conv.s,
        )
        self.first_conv = DynamicConvBnActLayer(
            in_channel_list=int2list(3),
            out_channel_list=out_channel_list,
            kernel_size=3,
            stride=stride,
            act_func=act_func,
        )

        # inverted residual blocks
        self.block_group_info = []
        blocks = []
        _block_index = 0
        feature_dim = out_channel_list
        for _, key in enumerate(self.stage_names[1:-1]):
            block_cfg = getattr(self.supernet, key)
            width = block_cfg.c
            n_block = max(block_cfg.d)
            act_func = block_cfg.act_func
            kernel_size = block_cfg.k
            expand_ratio_list = block_cfg.t
            use_se = block_cfg.se

            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                stride = block_cfg.s if i == 0 else 1
                if min(expand_ratio_list) >= 4:
                    expand_ratio_list = (
                        [_s for _s in expand_ratio_list if _s >= 4]
                        if i == 0
                        else expand_ratio_list
                    )
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=feature_dim,
                    out_channel_list=output_channel,
                    kernel_size_list=kernel_size,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func=act_func,
                    use_se=use_se,
                    channels_per_group=getattr(self.supernet, "channels_per_group", 1),
                )
                shortcut = DynamicShortcutLayer(
                    feature_dim, output_channel, reduction=stride
                )
                blocks.append(
                    MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)
                )
                feature_dim = output_channel
        self.blocks = nn.ModuleList(blocks)

        last_channel, act_func = (
            self.supernet.last_conv.c,
            self.supernet.last_conv.act_func,
        )
        expand_feature_dim = [f_dim * 6 for f_dim in feature_dim]
        self.last_conv = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "final_expand_layer",
                        DynamicConvBnActLayer(
                            feature_dim,
                            expand_feature_dim,
                            kernel_size=1,
                            use_bn=True,
                            act_func=act_func,
                        ),
                    ),
                    ("pool", nn.AdaptiveAvgPool2d((1, 1))),
                    (
                        "feature_mix_layer",
                        DynamicConvBnActLayer(
                            in_channel_list=expand_feature_dim,
                            out_channel_list=last_channel,
                            kernel_size=1,
                            act_func=act_func,
                            use_bn=False,
                        ),
                    ),
                ]
            )
        )

        # final conv layer
        self.classifier = DynamicLinearLayer(
            in_features_list=last_channel, out_features=n_classes, bias=True
        )

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

        self.zero_residual_block_bn_weights()

        self.active_dropout_rate = 0
        self.active_drop_connect_rate = 0
        self.active_resolution = 224

    def init_cfg_candidates(self):
        """Initialize the cfg candidates based on current supernet cfg."""
        self.use_v3_head = getattr(self.supernet, "use_v3_head", False)
        self.stage_names = [
            "first_conv",
            "mb1",
            "mb2",
            "mb3",
            "mb4",
            "mb5",
            "last_conv",
        ]

        self.width_list, self.depth_list, self.ks_list, self.expand_ratio_list = (
            [],
            [],
            [],
            [],
        )
        for name in self.stage_names:
            block_cfg = getattr(self.supernet, name)
            self.width_list.append(block_cfg.c)
            if name.startswith("mb"):
                self.depth_list.append(block_cfg.d)
                self.ks_list.append(block_cfg.k)
                self.expand_ratio_list.append(block_cfg.t)
        self.resolution_list = self.supernet.resolutions

        self.cfg_candidates = {
            "resolution": self.resolution_list,
            "width": self.width_list,
            "depth": self.depth_list,
            "kernel_size": self.ks_list,
            "expand_ratio": self.expand_ratio_list,
        }

    def get_active_net_config(self):
        """Get the config of current active net."""
        raise NotImplementedError
