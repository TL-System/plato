# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_

from .dynamic_ops import (
    DynamicBatchNorm2d,
    DynamicLinear,
    DynamicPointConv2d,
    DynamicSE,
    DynamicSeparableConv2d,
)
from .nn_base import MyModule, MyNetwork
from .nn_utils import (
    build_activation,
    copy_bn,
    get_net_device,
    int2list,
    make_divisible,
)
from .static_layers import (
    ConvBnActLayer,
    LinearLayer,
    MBInvertedConvLayer,
    SELayer,
    ShortcutLayer,
)


class DynamicMlp(MyModule):
    def __init__(self, hidden_features_list, out_features, bias=True, act_layer=None):
        super(DynamicMlp, self).__init__()
        self.hidden_features_list = int2list(hidden_features_list)
        self.out_features = out_features
        self.in_features = out_features
        self.bias = bias

        # self.bn1 = DynamicBatchNorm2d(self.in_features)
        # self.bn2 =  DynamicBatchNorm2d(max(self.hidden_features_list))
        self.linear1 = DynamicLinear(
            max_in_features=self.in_features,
            max_out_features=max(self.hidden_features_list),
            bias=self.bias,
        )
        self.linear2 = DynamicLinear(
            max_in_features=max(self.hidden_features_list),
            max_out_features=self.out_features,
            bias=self.bias,
        )
        self.act = build_activation(act_layer, inplace=True)
        self.active_hidden_features = max(self.hidden_features_list)

    def forward(self, x):
        B, N, C = x.shape
        C_ = C
        x = self.linear1(x, int(x.shape[-1] * 1.0))
        x = self.act(x)

        x = self.linear2(x, C_)
        return x

    @property
    def module_str(self):
        return "DyMLP(%d)" % self.out_features

    @property
    def config(self):
        return {
            "name": DynamicMLP.__name__,
            "hidden_features_list": self.hidden_features_list,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicMLP(**config)

    def get_active_subnet(self, hidden_features, preserve_weight=True):
        sub_layer = nn.Sequential(
            LinearLayer(self.in_features, hidden_features, self.bias),
            self.act,
            LinearLayer(hidden_features, self.out_features, self.bias),
        )
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer
        sub_layer[0].linear.weight.data.copy_(
            self.linear1.linear.weight.data[: self.in_features, :hidden_features]
        )
        sub_layer[-1].linear.weight.data.copy_(
            self.linear2.linear.weight.data[:hidden_features, : self.out_features]
        )
        if self.bias:
            sub_layer[0].linear.bias.data.copy_(
                self.linear1.linear.bias.data[:hidden_features]
            )
            sub_layer[-1].linear.bias.data.copy_(
                self.linear2.linear.bias.data[: self.out_features]
            )
        return sub_layer


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class OutlookAttention(MyModule):
    def __init__(
        self,
        dim_list,
        num_heads,
        kernel_size=3,
        padding=1,
        stride=1,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        act_layer="relu6",
        downsample=1,
    ):
        super(OutlookAttention, self).__init__()
        self.dim_list = dim_list
        self.head_dim = 8
        head_dim = self.head_dim
        self.dim = max(dim_list)
        dim = self.dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.expan_ratio = 1
        self.v = DynamicLinear(
            max_in_features=max(self.dim_list),
            max_out_features=max(self.dim_list) * self.expan_ratio,
            bias=qkv_bias,
        )
        self.attn = DynamicLinear(
            max_in_features=max(self.dim_list),
            max_out_features=kernel_size**4 * self.num_heads,
        )  # * self.expan_ratio)
        self.proj = DynamicLinear(
            max_in_features=max(self.dim_list) * self.expan_ratio,
            max_out_features=max(self.dim_list),
        )

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.softmax = nn.Softmax(dim=-1)
        self.act = build_activation(act_layer, inplace=True)

    def forward(self, x):
        B_, N, C = x.shape
        H = int(N**0.5)
        x = x.reshape(B_, H, H, C)
        B, H, W, C = x.shape
        v = self.v(x, out_features=x.shape[-1] * self.expan_ratio).permute(
            0, 3, 1, 2
        )  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = (
            self.unfold(v)
            .reshape(
                B,
                C // self.head_dim,
                self.expan_ratio * self.head_dim,
                self.kernel_size * self.kernel_size,
                h * w,
            )
            .permute(0, 1, 4, 3, 2)
        )  # b, exc, hxw, 3x3, num_head

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = (
            self.attn(
                attn,
                out_features=C
                // self.head_dim
                * self.kernel_size**4
                * self.expan_ratio,
            )
            .reshape(
                B,
                h * w,
                C // self.head_dim,
                self.kernel_size * self.kernel_size,
                self.kernel_size * self.kernel_size,
            )
            .permute(0, 2, 1, 3, 4)
        )  # b, c, hxw, 9, 9
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)

        x = (
            (attn @ v)
            .permute(0, 1, 4, 3, 2)
            .reshape(
                B, C * self.kernel_size * self.kernel_size * self.expan_ratio, h * w
            )
        )
        x = F.fold(
            x,
            output_size=(H, W),
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        x = self.act(x).permute(0, 2, 3, 1).reshape(B, N, -1)
        x = self.proj(x, out_features=x.shape[-1] // self.expan_ratio)
        return x


class DynamicWindowAttention(MyModule):
    def __init__(
        self,
        dim_list,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        act_layer="relu6",
        downsample=1,
    ):
        super(DynamicWindowAttention, self).__init__()

        self.dim_list = dim_list
        # self.head_dim = max(dim_list) // num_heads
        self.head_dim = 8

        head_dim = self.head_dim
        self.dim = max(dim_list)
        self.rpe_bias = self.dim // self.head_dim

        dim = self.dim
        # self.num_heads = num_heads
        self.num_heads = self.dim // head_dim
        self.window_size = window_size  # Wh, Ww
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads * 1
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.expan_ratio = 4  # 2
        self.q = DynamicLinear(
            max_in_features=max(self.dim_list),
            max_out_features=max(self.dim_list),
            bias=qkv_bias,
        )
        self.k = DynamicLinear(
            max_in_features=max(self.dim_list),
            max_out_features=max(self.dim_list),
            bias=qkv_bias,
        )
        self.v = DynamicLinear(
            max_in_features=max(self.dim_list),
            max_out_features=max(self.dim_list) * self.expan_ratio,
            bias=qkv_bias,
        )

        self.vconv = nn.Conv2d(
            dim * self.expan_ratio,
            dim * self.expan_ratio,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * self.expan_ratio,
            bias=False,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = DynamicLinear(
            max_in_features=max(self.dim_list) * self.expan_ratio,
            max_out_features=max(self.dim_list),
        )
        self.proj_drop = nn.Dropout(proj_drop)

        self.proj_l = DynamicLinear(
            max_in_features=max(self.dim_list) // self.head_dim,
            max_out_features=max(self.dim_list) // self.head_dim,
        )
        self.proj_w = DynamicLinear(
            max_in_features=max(self.dim_list) // self.head_dim,
            max_out_features=max(self.dim_list) // self.head_dim,
        )

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
        self.act = build_activation(act_layer, inplace=True)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = (
            self.q(x, out_features=x.shape[-1])
            .reshape(B_, N, C // self.head_dim, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x, out_features=x.shape[-1])
            .reshape(B_, N, C // self.head_dim, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        v = self.v(x, out_features=x.shape[-1] * self.expan_ratio)
        v = F.conv2d(
            v.permute(0, 2, 1).reshape(B_, -1, int(N**0.5), int(N**0.5)),
            self.vconv.weight[: C * self.expan_ratio, :, :, :],
            None,
            1,
            1,
            1,
            C * self.expan_ratio,
        )
        v = (
            v.reshape(B_, -1, N)
            .permute(0, 2, 1)
            .reshape(B_, N, C // self.head_dim, -1)
            .permute(0, 2, 1, 3)
        )
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0],
            self.window_size[1],
            self.window_size[0],
            self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        H, W = int(attn.shape[3] ** 0.5), int(attn.shape[3] ** 0.5)

        relative_position_bias = (
            relative_position_bias[:H, :W, :H, :W, :]
            .contiguous()
            .reshape(H * W, H * W, -1)
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = (
            attn + relative_position_bias.unsqueeze(0)[:, : attn.shape[1]]
        )  # , :attn.shape[2], :attn.shape[3]]

        attn = self.proj_l(
            attn.permute(0, 2, 3, 1), out_features=C // self.head_dim
        ).permute(0, 3, 1, 2)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.proj_w(
            attn.permute(0, 2, 3, 1), out_features=C // self.head_dim
        ).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(self.act(x), out_features=x.shape[-1] // self.expan_ratio)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}"  # , num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

    @property
    def module_str(self):
        return "DyMLP(%d)" % self.out_features

    @property
    def config(self):
        return {
            "name": DynamicWindowAttention.__name__,
            "dim_list": self.dim_list,
            "window_size": self.window_size,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicWindowAttention(**config)

    def get_active_subnet(self, dim, preserve_weight=True):
        sub_layer = WindowAttention(dim, self.window_size, dim // 16)
        sub_layer = sub_layer.to(get_net_device(self))
        sub_layer.qkv.weight.data.copy_(self.qkv.weight.data[:dim, : dim * 6])
        sub_layer.proj.weight.data.copy_(self.proj.weight.data[: dim * 4, :dim])
        return sub_layer


class DynamicSwinTransformerBlock(MyModule):
    def __init__(
        self,
        dim_list,
        input_resolution,
        num_heads,
        window_size=14,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        downsample=1,
        rescale=1.0,
        shift=False,
    ):
        super(DynamicSwinTransformerBlock, self).__init__()
        self.dim_list = dim_list
        self.dim = max(dim_list)
        self.rescale = rescale
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.shift = shift

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, (
            "shift_size must in 0-window_size"
        )

        self.norm1 = norm_layer(self.dim)
        self.attn = DynamicWindowAttention(
            dim_list,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            act_layer=act_layer,
            downsample=downsample,
        )

        self.drop_path = DropPath(drop_path)  # if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)

        self.norm3 = norm_layer(self.dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp1 = DynamicMlp(
            out_features=self.dim,
            hidden_features_list=[int(1.0 * self.dim)],
            act_layer=act_layer,
        )  # , drop=drop)
        self.mlp2 = DynamicMlp(
            out_features=self.dim,
            hidden_features_list=[int(1.0 * self.dim)],
            act_layer=act_layer,
        )  # , drop=drop)
        self.mobile_inverted_conv = None

        self.rescale_mlp = nn.Parameter(
            1e-4 * torch.ones((10, 8, self.dim)), requires_grad=True
        )
        self.rescale_attn = nn.Parameter(
            1e-4 * torch.ones((10, 8, self.dim)), requires_grad=True
        )

        self.rescale_idx = 0

    def forward(self, x):
        # H, W = self.input_resolution
        # x = self.downsample(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = F.layer_norm(
            x,
            x.shape[1:],
            self.norm1.weight[: x.shape[-1]].expand(x.shape[1:]),
            self.norm1.bias[: x.shape[-1]].expand(x.shape[1:]),
        )

        if H > 18 and self.shift:
            if H > 18:
                num_window = 2
            else:
                num_window = 2
            x = x.view(B, H, W, C)
            if self.shift:
                self.shift_size = H // num_window // 2
                shifted_x = torch.roll(
                    x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )
            else:
                shifted_x = x

            x_windows = window_partition(shifted_x, H // num_window)
            x_windows = x_windows.view(-1, H // num_window * H // num_window, C)
            attn_windows = self.attn(x_windows)  # , mask=self.attn_mask)
            attn_windows = attn_windows.view(-1, H // num_window * H // num_window, C)
            shifted_x = window_reverse(attn_windows, H // num_window, H, W)
            if self.shift:
                x = torch.roll(
                    shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                )
            else:
                x = shifted_x
            x = x.view(B, H * W, C)
        else:
            x = self.attn(x)

        self.rescale_idx = self.rescale_idx
        self.rescale_idx_width = x.shape[1] % 8

        # FFN
        x = shortcut + self.drop_path(
            x
            * self.rescale_attn[self.rescale_idx, self.rescale_idx_width, :C]
            * self.rescale_mlp.shape[1]
            / C
        )
        x = x + self.drop_path(
            self.mlp2(
                F.layer_norm(
                    x,
                    x.shape[1:],
                    self.norm3.weight[: x.shape[-1]].expand(x.shape[1:]),
                    self.norm3.bias[: x.shape[-1]].expand(x.shape[1:]),
                )
            )
            * self.rescale_mlp[self.rescale_idx, self.rescale_idx_width, :C]
            * self.rescale_mlp.shape[1]
            / C
        )
        x = x + self.drop_path(
            self.mlp1(
                F.layer_norm(
                    x,
                    x.shape[1:],
                    self.norm2.weight[: x.shape[-1]].expand(x.shape[1:]),
                    self.norm2.bias[: x.shape[-1]].expand(x.shape[1:]),
                )
            )
            * self.rescale_mlp[self.rescale_idx, self.rescale_idx_width, :C]
            * self.rescale_mlp.shape[1]
            / C
        )
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class DynamicPatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super(DynamicPatchMerging, self).__init__()
        self.dim = dim
        self.reduction = DynamicLinear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.mobile_inverted_conv = None
        self.out_dim = out_dim

    def forward(self, x):
        B, C, H, W = x.shape
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, W % 2, 0, H % 2))
        B, C, pH, pW = x.shape
        x = (
            x.reshape(B, C, pH // 2, 2, pW // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, 4 * C, -1)
            .permute(0, 2, 1)
        )

        x = F.layer_norm(
            x,
            x.shape[1:],
            self.norm.weight[: x.shape[-1]].expand(x.shape[1:]),
            self.norm.bias[: x.shape[-1]].expand(x.shape[1:]),
        )
        # x = self.norm(x)
        x = self.reduction(x, self.out_dim)
        N = x.shape[1]
        return (
            x.reshape(B, -1, self.out_dim)
            .permute(0, 2, 1)
            .reshape(B, self.out_dim, int(N**0.5), int(N**0.5))
        )

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


# class BasicLayer(nn.Module):
#     """ A basic Swin Transformer layer for one stage.

#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """

#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint

#         # build blocks
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
#                                  num_heads=num_heads, window_size=window_size,
#                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                  mlp_ratio=mlp_ratio,
#                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                  drop=drop, attn_drop=attn_drop,
#                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                  norm_layer=norm_layer)
#             for i in range(depth)])

#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None

#     def forward(self, x):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x

#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops


# class PatchEmbed(nn.Module):
#     r""" Image to Patch Embedding

#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """

#     def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
#         if self.norm is not None:
#             x = self.norm(x)
#         return x

#     def flops(self):
#         Ho, Wo = self.patches_resolution
#         flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
#         if self.norm is not None:
#             flops += Ho * Wo * self.embed_dim
#         return flops


# class SwinTransformer(nn.Module):
#     r""" Swin Transformer
#         A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
#           https://arxiv.org/pdf/2103.14030

#     Args:
#         img_size (int | tuple(int)): Input image size. Default 224
#         patch_size (int | tuple(int)): Patch size. Default: 4
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         embed_dim (int): Patch embedding dimension. Default: 96
#         depths (tuple(int)): Depth of each Swin Transformer layer.
#         num_heads (tuple(int)): Number of attention heads in different layers.
#         window_size (int): Window size. Default: 7
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
#         drop_rate (float): Dropout rate. Default: 0
#         attn_drop_rate (float): Attention dropout rate. Default: 0
#         drop_path_rate (float): Stochastic depth rate. Default: 0.1
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
#     """

#     def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
#                  embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
#                  window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, **kwargs):
#         super().__init__()

#         self.num_classes = num_classes
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
#         self.mlp_ratio = mlp_ratio

#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution

#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)

#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

#         # build layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
#                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
#                                                  patches_resolution[1] // (2 ** i_layer)),
#                                depth=depths[i_layer],
#                                num_heads=num_heads[i_layer],
#                                window_size=window_size,
#                                mlp_ratio=self.mlp_ratio,
#                                qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                drop=drop_rate, attn_drop=attn_drop_rate,
#                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                                norm_layer=norm_layer,
#                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                                use_checkpoint=use_checkpoint)
#             self.layers.append(layer)

#         self.norm = norm_layer(self.num_features)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}

#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}

#     def forward_features(self, x):
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)

#         count = 0
#         layers = []
#         for layer in self.layers:
#             x = layer(x)
#             if count == 1 or count == 2:
#                 layers.append(x)
#             count += 1


#         x = self.norm(x)  # B L C
#         layers.append(x * torch.ones_like(x))
#         x = self.avgpool(x.transpose(1, 2))  # B C 1
#         x = torch.flatten(x, 1)
#         return x, layers

#     def forward(self, x):
#         x, layers_outputs = self.forward_features(x)

#         x = self.head(x)
#         input()
#         if self.training:
#             return (x, x), layers_outputs
#         return x, layers_outputs

#     def flops(self):
#         flops = 0
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.layers):
#             flops += layer.flops()
#         flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
#         flops += self.num_features * self.num_classes
#         return flops
