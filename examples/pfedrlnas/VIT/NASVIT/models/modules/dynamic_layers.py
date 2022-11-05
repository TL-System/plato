# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# modified from OFA: https://github.com/mit-han-lab/once-for-all

from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .static_layers import (
    MBInvertedConvLayer,
    ConvBnActLayer,
    LinearLayer,
    SELayer,
    ShortcutLayer,
)
from .dynamic_ops import (
    DynamicSeparableConv2d,
    DynamicPointConv2d,
    DynamicBatchNorm2d,
    DynamicLinear,
    DynamicSE,
)
from .nn_utils import (
    int2list,
    get_net_device,
    copy_bn,
    build_activation,
    make_divisible,
)
from .nn_base import MyModule, MyNetwork


class DynamicMBConvLayer(MyModule):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size_list=3,
        expand_ratio_list=6,
        stride=1,
        act_func="relu6",
        use_se=False,
        channels_per_group=1,
    ):
        super(DynamicMBConvLayer, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)

        self.kernel_size_list = int2list(kernel_size_list, 1)
        self.expand_ratio_list = int2list(expand_ratio_list, 1)

        self.stride = stride
        self.act_func = act_func
        self.use_se = use_se
        self.channels_per_group = channels_per_group

        # build modules
        max_middle_channel = round(
            max(self.in_channel_list) * max(self.expand_ratio_list)
        )
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            DynamicPointConv2d(
                                max(self.in_channel_list), max_middle_channel
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max_middle_channel)),
                        ("act", build_activation(self.act_func, inplace=True)),
                    ]
                )
            )

        self.depth_conv = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicSeparableConv2d(
                            max_middle_channel,
                            self.kernel_size_list,
                            stride=self.stride,
                            channels_per_group=self.channels_per_group,
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max_middle_channel)),
                    ("act", build_activation(self.act_func, inplace=True)),
                ]
            )
        )
        if self.use_se:
            self.depth_conv.add_module("se", DynamicSE(max_middle_channel))

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicPointConv2d(
                            max_middle_channel, max(self.out_channel_list)
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                ]
            )
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

        self.rescale = nn.Parameter(
            1.0 * torch.ones([max(self.out_channel_list)]), requires_grad=False
        )
        # self.rescale = nn.Parameter(1e-2 * torch.ones([10, max(self.out_channel_list)]), requires_grad=True)
        self.rescale_idx = 0

    def forward(self, x):
        in_channel = x.size(1)

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = make_divisible(
                round(in_channel * self.active_expand_ratio), 8
            )

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)

        # if x.shape[-1] % 2 != 0:
        #     x = F.pad(x, (0, 1, 0, 1), "constant", 0)

        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x  # * float(max(self.out_channel_list)) / x.shape[1] # / (self.rescale_idx + 1.) # self.rescale[:x.shape[1]].reshape([1, -1, 1, 1])

    @property
    def module_str(self):
        if self.use_se:
            return "SE(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )
        else:
            return "(O%d, E%.1f, K%d)" % (
                self.active_out_channel,
                self.active_expand_ratio,
                self.active_kernel_size,
            )

    @property
    def config(self):
        return {
            "name": DynamicMBConvLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size_list": self.kernel_size_list,
            "expand_ratio_list": self.expand_ratio_list,
            "stride": self.stride,
            "act_func": self.act_func,
            "use_se": self.use_se,
            "channels_per_group": self.channels_per_group,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicMBConvLayer(**config)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        middle_channel = make_divisible(round(in_channel * self.active_expand_ratio), 8)
        channels_per_group = self.depth_conv.conv.channels_per_group

        # build the new layer
        sub_layer = MBInvertedConvLayer(
            in_channel,
            self.active_out_channel,
            self.active_kernel_size,
            self.stride,
            self.active_expand_ratio,
            act_func=self.act_func,
            mid_channels=middle_channel,
            use_se=self.use_se,
            channels_per_group=channels_per_group,
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.conv.weight.data[
                    :middle_channel, :in_channel, :, :
                ]
            )
            copy_bn(sub_layer.inverted_bottleneck.bn, self.inverted_bottleneck.bn.bn)

        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(
                middle_channel, self.active_kernel_size
            ).data
        )

        # sub_layer.rescale.data.copy_(self.rescale.data[:self.active_out_channel])
        sub_layer.rescale = 1.0  # float(max(self.out_channel_list)) / self.active_out_channel # * (self.rescale_idx + 1.)# .data.copy_(self.rescale.data[self.rescale_idx, :self.active_out_channel])
        copy_bn(sub_layer.depth_conv.bn, self.depth_conv.bn.bn)

        if self.use_se:
            se_mid = make_divisible(middle_channel // SELayer.REDUCTION, divisor=8)
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.fc.reduce.weight.data[:se_mid, :middle_channel, :, :]
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(
                self.depth_conv.se.fc.reduce.bias.data[:se_mid]
            )

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.fc.expand.weight.data[:middle_channel, :se_mid, :, :]
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(
                self.depth_conv.se.fc.expand.bias.data[:middle_channel]
            )

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.conv.weight.data[
                : self.active_out_channel, :middle_channel, :, :
            ]
        )
        copy_bn(sub_layer.point_linear.bn, self.point_linear.bn.bn)

        return sub_layer

    ############################################################################################

    def get_weight_from_subnet(self, in_channel, sub_layer):
        """Get weight from subnet of this basic dynamic operation."""
        middle_channel = make_divisible(round(in_channel * self.active_expand_ratio), 8)
        # channels_per_group = self.depth_conv.conv.channels_per_group

        # copy weight from current layer
        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.conv.weight.data[
                :middle_channel, :in_channel, :, :
            ].copy_(sub_layer.inverted_bottleneck.conv.weight.data)
            copy_bn(self.inverted_bottleneck.bn.bn, sub_layer.inverted_bottleneck.bn)

        self.depth_conv.conv.get_active_filter(
            middle_channel, self.active_kernel_size
        ).data.copy_(sub_layer.depth_conv.conv.weight.data)

        # sub_layer.rescale.data.copy_(self.rescale.data[:self.active_out_channel])
        sub_layer.rescale = 1.0  # float(max(self.out_channel_list)) / self.active_out_channel # * (self.rescale_idx + 1.)# .data.copy_(self.rescale.data[self.rescale_idx, :self.active_out_channel])
        copy_bn(self.depth_conv.bn.bn, sub_layer.depth_conv.bn)

        if sub_layer.use_se:
            se_mid = make_divisible(middle_channel // SELayer.REDUCTION, divisor=8)
            self.depth_conv.se.fc.reduce.weight.data[
                :se_mid, :middle_channel, :, :
            ].copy_(sub_layer.depth_conv.se.fc.reduce.weight.data)
            self.depth_conv.se.fc.reduce.bias.data[:se_mid].copy_(
                sub_layer.depth_conv.se.fc.reduce.bias.data
            )

            self.depth_conv.se.fc.expand.weight.data[
                :middle_channel, :se_mid, :, :
            ].copy_(sub_layer.depth_conv.se.fc.expand.weight.data)
            self.depth_conv.se.fc.expand.bias.data[:middle_channel].copy_(
                sub_layer.depth_conv.se.fc.expand.bias.data
            )

        self.point_linear.conv.conv.weight.data[
            : self.active_out_channel, :middle_channel, :, :
        ].copy_(sub_layer.point_linear.conv.weight.data)
        copy_bn(self.point_linear.bn.bn, sub_layer.point_linear.bn)

        return sub_layer

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        raise NotImplementedError
        # importance = torch.sum(torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2, 3))
        # if expand_ratio_stage > 0:
        #    sorted_expand_list = copy.deepcopy(self.expand_ratio_list)
        #    sorted_expand_list.sort(reverse=True)
        #    target_width = sorted_expand_list[expand_ratio_stage]
        #    target_width = round(max(self.in_channel_list) * target_width)
        #    importance[target_width:] = torch.arange(0, target_width - importance.size(0), -1)
        #
        # sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        # self.point_linear.conv.conv.weight.data = torch.index_select(
        #    self.point_linear.conv.conv.weight.data, 1, sorted_idx
        # )
        #
        # adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        # self.depth_conv.conv.conv.weight.data = torch.index_select(
        #    self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        # )

        # if self.use_se:
        #    # se expand: output dim 0 reorganize
        #    se_expand = self.depth_conv.se.fc.expand
        #    se_expand.weight.data = torch.index_select(se_expand.weight.data, 0, sorted_idx)
        #    se_expand.bias.data = torch.index_select(se_expand.bias.data, 0, sorted_idx)
        #    # se reduce: input dim 1 reorganize
        #    se_reduce = self.depth_conv.se.fc.reduce
        #    se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 1, sorted_idx)
        #    # middle weight reorganize
        #    se_importance = torch.sum(torch.abs(se_expand.weight.data), dim=(0, 2, 3))
        #    se_importance, se_idx = torch.sort(se_importance, dim=0, descending=True)

        #    se_expand.weight.data = torch.index_select(se_expand.weight.data, 1, se_idx)
        #    se_reduce.weight.data = torch.index_select(se_reduce.weight.data, 0, se_idx)
        #    se_reduce.bias.data = torch.index_select(se_reduce.bias.data, 0, se_idx)
        #
        ## TODO if inverted_bottleneck is None, the previous layer should be reorganized accordingly
        # if self.inverted_bottleneck is not None:
        #    adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
        #    self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
        #        self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
        #    )
        #    return None
        # else:
        #    return sorted_idx


class DynamicConvBnActLayer(MyModule):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size=3,
        stride=1,
        dilation=1,
        use_bn=True,
        act_func="relu6",
    ):
        super(DynamicConvBnActLayer, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        self.conv = DynamicPointConv2d(
            max_in_channels=max(self.in_channel_list),
            max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))

        if self.act_func is not None:
            self.act = build_activation(self.act_func, inplace=True)

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act_func is not None:
            x = self.act(x)
        return x

    @property
    def module_str(self):
        return "DyConv(O%d, K%d, S%d)" % (
            self.active_out_channel,
            self.kernel_size,
            self.stride,
        )

    @property
    def config(self):
        return {
            "name": DynamicConvBnActLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicConvBnActLayer(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = ConvBnActLayer(
            in_channel,
            self.active_out_channel,
            self.kernel_size,
            self.stride,
            self.dilation,
            use_bn=self.use_bn,
            act_func=self.act_func,
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(
            self.conv.conv.weight.data[: self.active_out_channel, :in_channel, :, :]
        )
        if self.use_bn:
            copy_bn(sub_layer.bn, self.bn.bn)

        return sub_layer

    def get_weight_from_subnet(self, in_channel, sub_layer):
        """Get weight from subnet of this basic dynamic operation."""
        self.conv.conv.weight.data[: self.active_out_channel, :in_channel, :, :].copy_(
            sub_layer.conv.weight.data
        )
        if sub_layer.use_bn:
            copy_bn(self.bn.bn, sub_layer.bn)

        return sub_layer


class DynamicLinearLayer(MyModule):
    def __init__(self, in_features_list, out_features, bias=True):
        super(DynamicLinearLayer, self).__init__()

        self.in_features_list = int2list(in_features_list)
        self.out_features = out_features
        self.bias = bias
        # self.dropout_rate = dropout_rate
        #
        # if self.dropout_rate > 0:
        #    self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        # else:
        #    self.dropout = None
        self.linear = DynamicLinear(
            max_in_features=max(self.in_features_list),
            max_out_features=self.out_features,
            bias=self.bias,
        )

    def forward(self, x):
        # if self.dropout is not None:
        #    x = self.dropout(x)
        return self.linear(x)

    @property
    def module_str(self):
        return "DyLinear(%d)" % self.out_features

    @property
    def config(self):
        return {
            "name": DynamicLinear.__name__,
            "in_features_list": self.in_features_list,
            "out_features": self.out_features,
            "bias": self.bias,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicLinearLayer(**config)

    def get_active_subnet(self, in_features, preserve_weight=True):
        # sub_layer = LinearLayer(in_features, self.out_features, self.bias, dropout_rate=self.dropout_rate)
        sub_layer = LinearLayer(in_features, self.out_features, self.bias)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        sub_layer.linear.weight.data.copy_(
            self.linear.linear.weight.data[: self.out_features, :in_features]
        )
        if self.bias:
            sub_layer.linear.bias.data.copy_(
                self.linear.linear.bias.data[: self.out_features]
            )
        return sub_layer

    def get_weight_from_subnet(self, in_features, sub_layer):
        """Get weight from subnet of this basic dynamic operation."""
        self.linear.linear.weight.data[: self.out_features, :in_features].copy_(
            sub_layer.linear.weight.data
        )
        if sub_layer.bias:
            self.linear.linear.bias.data[: self.out_features].copy_(
                sub_layer.linear.bias.data
            )
        return sub_layer


class DynamicShortcutLayer(MyModule):
    def __init__(self, in_channel_list, out_channel_list, reduction=1):
        super(DynamicShortcutLayer, self).__init__()

        self.in_channel_list = int2list(in_channel_list)
        self.out_channel_list = int2list(out_channel_list)
        self.reduction = reduction

        self.conv = DynamicPointConv2d(
            max_in_channels=max(self.in_channel_list),
            max_out_channels=max(self.out_channel_list),
            kernel_size=1,
            stride=1,
        )

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        in_channel = x.size(1)

        # identity mapping
        if in_channel == self.active_out_channel and self.reduction == 1:
            return x

        # change: cygong

        # average pooling, if size doesn't match
        if self.reduction > 1:
            padding = 0 if x.size(-1) % 2 == 0 else 1
            x = F.avg_pool2d(x, self.reduction, padding=padding)
            # x = torch.zeros_like(F.avg_pool2d(x, self.reduction, padding=padding))

        # 1*1 conv, if #channels doesn't match
        if in_channel != self.active_out_channel:
            self.conv.active_out_channel = self.active_out_channel
            x = self.conv(x)
            # x = torch.zeros_like(self.conv(x))

        return x

    @property
    def module_str(self):
        return "DyShortcut(O%d, R%d)" % (self.active_out_channel, self.reduction)

    @property
    def config(self):
        return {
            "name": DynamicShortcutLayer.__name__,
            "in_channel_list": self.in_channel_list,
            "out_channel_list": self.out_channel_list,
            "reduction": self.reduction,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicShortcutLayer(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        sub_layer = ShortcutLayer(in_channel, self.active_out_channel, self.reduction)
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(
            self.conv.conv.weight.data[: self.active_out_channel, :in_channel, :, :]
        )

        return sub_layer

    def get_weight_from_subnet(self, in_channel, sub_layer):
        """Get weight from subnet of this basic dynamic operation."""
        self.conv.conv.weight.data[: self.active_out_channel, :in_channel, :, :].copy_(
            sub_layer.conv.weight.data
        )

        return sub_layer
