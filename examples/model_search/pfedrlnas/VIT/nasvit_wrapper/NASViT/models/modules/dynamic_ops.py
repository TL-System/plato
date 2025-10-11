# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# modified from OFA: https://github.com/mit-han-lab/once-for-all

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
from torch.nn.parameter import Parameter

from .nn_utils import get_same_padding, make_divisible, sub_filter_start_end
from .static_layers import SELayer


class DynamicSeparableConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = None  # None or 1

    def __init__(
        self,
        max_in_channels,
        kernel_size_list,
        stride=1,
        dilation=1,
        channels_per_group=1,
    ):
        super(DynamicSeparableConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.channels_per_group = channels_per_group
        assert self.max_in_channels % self.channels_per_group == 0
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels // self.channels_per_group,
            bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = "%dto%d" % (ks_larger, ks_small)
                scale_params["%s_matrix" % param_name] = Parameter(
                    torch.eye(ks_small**2)
                )
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[
                :out_channel, :in_channel, :, :
            ]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1
                )
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks**2
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)
        assert in_channel % self.channels_per_group == 0

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()

        padding = get_same_padding(kernel_size)
        y = F.conv2d(
            x,
            filters,
            None,
            self.stride,
            padding,
            self.dilation,
            in_channel // self.channels_per_group,
        )
        return y


class DynamicPointConv2d(nn.Module):
    def __init__(
        self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1
    ):
        super(DynamicPointConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.conv.weight[:out_channel, :in_channel, :, :].contiguous()

        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicLinear(nn.Module):
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(-1)
        weight = self.linear.weight[:out_features, :in_features].contiguous()
        bias = self.linear.bias[:out_features] if self.bias else None
        y = F.linear(x, weight, bias)
        return y


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class DynamicBatchNorm2d(nn.Module):
    """
    1. doesn't acculate bn statistics, (momentum=0.)
    2. calculate BN statistics of all subnets after training
    3. bn weights are shared
    https://arxiv.org/abs/1903.05134
    https://detectron2.readthedocs.io/_modules/detectron2/layers/batch_norm.html
    """

    # SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super(DynamicBatchNorm2d, self).__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

        # self.exponential_average_factor = 0 #doesn't acculate bn stats
        self.need_sync = False

        # reserved to tracking the performance of the largest and smallest network
        self.bn_tracking = nn.ModuleList(
            [
                nn.BatchNorm2d(self.max_feature_dim, affine=False),
                nn.BatchNorm2d(self.max_feature_dim, affine=False),
            ]
        )

    def forward(self, x):
        feature_dim = x.size(1)
        if not self.training:
            raise ValueError("DynamicBN only supports training")

        bn = self.bn
        # need_sync
        if not self.need_sync:
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                bn.momentum,
                bn.eps,
            )
        else:
            assert dist.get_world_size() > 1, "SyncBatchNorm requires >1 world size"
            B, C = x.shape[0], x.shape[1]
            mean = torch.mean(x, dim=[0, 2, 3])
            meansqr = torch.mean(x * x, dim=[0, 2, 3])
            assert B > 0, "does not support zero batch size"
            vec = torch.cat([mean, meansqr], dim=0)
            vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)

            var = meansqr - mean * mean
            invstd = torch.rsqrt(var + bn.eps)
            scale = bn.weight[:feature_dim] * invstd
            bias = bn.bias[:feature_dim] - mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias

        # if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
        #    return bn(x)
        # else:
        #    exponential_average_factor = 0.0

        #    if bn.training and bn.track_running_stats:
        #        # TODO: if statement only here to tell the jit to skip emitting this when it is None
        #        if bn.num_batches_tracked is not None:
        #            bn.num_batches_tracked += 1
        #            if bn.momentum is None:  # use cumulative moving average
        #                exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
        #            else:  # use exponential moving average
        #                exponential_average_factor = bn.momentum
        #    return F.batch_norm(
        #        x, bn.running_mean[:feature_dim], bn.running_var[:feature_dim], bn.weight[:feature_dim],
        #        bn.bias[:feature_dim], bn.training or not bn.track_running_stats,
        #        exponential_average_factor, bn.eps,
        #    )


class DynamicSE(SELayer):
    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)

    def forward(self, x):
        in_channel = x.size(1)
        num_mid = make_divisible(in_channel // self.reduction, divisor=8)

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_conv = self.fc.reduce
        reduce_filter = reduce_conv.weight[:num_mid, :in_channel, :, :].contiguous()
        reduce_bias = (
            reduce_conv.bias[:num_mid] if reduce_conv.bias is not None else None
        )
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_conv = self.fc.expand
        expand_filter = expand_conv.weight[:in_channel, :num_mid, :, :].contiguous()
        expand_bias = (
            expand_conv.bias[:in_channel] if expand_conv.bias is not None else None
        )
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)

        return x * y
