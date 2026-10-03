"""
Inherit modules fron NASViT and add get weights from subnet methods.
"""

from .NASViT.models.modules import dynamic_layers
from .NASViT.models.modules.nn_utils import make_divisible
from .NASViT.models.modules.static_layers import SELayer


def copy_bn(target_bn, src_bn):
    "Copy the BN with exact size."
    feature_dim = min(target_bn.num_features, src_bn.num_features)

    target_bn.weight.data[:feature_dim].copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data[:feature_dim].copy_(src_bn.bias.data[:feature_dim])
    target_bn.running_mean.data[:feature_dim].copy_(
        src_bn.running_mean.data[:feature_dim]
    )
    target_bn.running_var.data[:feature_dim].copy_(
        src_bn.running_var.data[:feature_dim]
    )


# pylint:disable=abstract-method
class DynamicMBConvLayer(dynamic_layers.DynamicMBConvLayer):
    """
    Added get weight from subnet.
    """

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

        sub_layer.rescale = 1.0
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


class DynamicConvBnActLayer(dynamic_layers.DynamicConvBnActLayer):
    """
    Added get weight from subnet.
    """

    def get_weight_from_subnet(self, in_channel, sub_layer):
        """Get weight from subnet of this basic dynamic operation."""
        self.conv.conv.weight.data[: self.active_out_channel, :in_channel, :, :].copy_(
            sub_layer.conv.weight.data
        )
        if sub_layer.use_bn:
            copy_bn(self.bn.bn, sub_layer.bn)

        return sub_layer


class DynamicLinearLayer(dynamic_layers.DynamicLinearLayer):
    """
    Added get weight from subnet.
    """

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


class DynamicShortcutLayer(dynamic_layers.DynamicShortcutLayer):
    """
    Added get weight from subnet.
    """

    def get_weight_from_subnet(self, in_channel, sub_layer):
        """Get weight from subnet of this basic dynamic operation."""
        self.conv.conv.weight.data[: self.active_out_channel, :in_channel, :, :].copy_(
            sub_layer.conv.weight.data
        )

        return sub_layer
