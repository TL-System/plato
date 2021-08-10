#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.models import backbones
from mmaction.models import heads
from mmaction.models import losses

from mmaction.models import build_model

from plato.models.multimodal import build_fc_from_config


class BaseClassificationNet(nn.Module):
    def __init__(self, net_configs, is_head_included=True):
        super(BaseClassificationNet, self).__init__()

        self.net_configs = net_configs
        # 1 build the model based on the configurations
        self._net = build_model(net_configs)

    def forward_train(self, ipt_data, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        outputs = []

        ipt_data = ipt_data.reshape((-1, ) + ipt_data.shape[2:])
        # 1. forward the backbone
        data_feat = self._net.extract_feat(ipt_data)

        # 2. forward the classification head if possible and obtain the losses
        loss_cls = 0.0
        if is_head_included:
            cls_score = self._net.cls_head(data_feat)

            gt_labels = labels.squeeze()
            loss_cls = self._net.cls_head.loss(cls_score, gt_labels, **kwargs)

            return [data_feat, loss_cls]

        return [data_feat, _]

    def forward_test(self, ipt_data, **kwargs):
        """Defines the computation performed at every call when training."""

        ipt_data = ipt_data.reshape((-1, ) + ipt_data.shape[2:])
        # 1. forward the backbone
        data_feat = self._net.extract_feat(ipt_data)
        # 2. forward the classification head if possible and obtain the losses
        cls_score = 0.0
        if is_head_included:
            cls_score = self._net.cls_head(data_feat)

            return [data_feat, cls_score]

        return [data_feat, _]

    def forward(self,
                ipt_data
                label=None,
                return_loss=True,
                **kwargs):
        """Defines the computation performed at every call.

        Args:
            ipt_data (torch.Tensor): The input data.
                the size of x is (num_batches, channel, num_slices, h, w).
        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            if self._net.blending is not None:
                blended_ipt_data, label = self._net.blending(ipt_data, label)


            return self.forward_train(blended_ipt_data, 
                                      label, **kwargs)

        return self.forward_test(blended_ipt_data, **kwargs)