# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# modified from OFA: https://github.com/mit-han-lab/once-for-all

import math

import torch
import torch.nn as nn

try:
    from fvcore.common.file_io import PathManager
except:
    pass


class MyModule(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class MyNetwork(MyModule):
    def forward(self, x):
        raise NotImplementedError

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def zero_last_gamma(self):
        raise NotImplementedError

    """ implemented methods """

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                if momentum is not None:
                    m.momentum = float(momentum)
                else:
                    m.momentum = None
                m.eps = float(eps)
        return

    def get_bn_param(self):
        for m in self.modules():
            if (
                isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
                or isinstance(m, nn.SyncBatchNorm)
            ):
                return {
                    "momentum": m.momentum,
                    "eps": m.eps,
                }
        return None

    def init_model(self, model_init):
        """Conv2d, BatchNorm2d, BatchNorm1d, Linear,"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == "he_fout":
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif model_init == "he_fin":
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                else:
                    raise NotImplementedError
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1.0 / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_parameters(self, keys=None, mode="include", exclude_set=None):
        if exclude_set is None:
            exclude_set = {}
        if keys is None:
            for name, param in self.named_parameters():
                if name not in exclude_set:
                    yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and name not in exclude_set:
                    yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and name not in exclude_set:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)

    def weight_parameters(self, exclude_set=None):
        return self.get_parameters(exclude_set=exclude_set)

    def load_weights_from_pretrained_models(self, checkpoint_path, load_from_ema=False):
        try:
            with PathManager.open(checkpoint_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
        except:
            with open(checkpoint_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
        assert isinstance(checkpoint, dict)
        pretrained_state_dicts = checkpoint["state_dict"]
        if load_from_ema and "state_dict_ema" in checkpoint:
            pretrained_state_dicts = checkpoint["state_dict_ema"]
        for k, v in self.state_dict().items():
            name = k
            if not load_from_ema:
                name = "module." + k if not k.startswith("module") else k
            v.copy_(pretrained_state_dicts[name])
