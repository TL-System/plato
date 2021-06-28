#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-06-27 13:23:05
"""
This multimodal network is the core network used in our paper. 
    It can receives three datasets from three modalities(RGB, optical flow, and audio) and then process them with 
three different networks:
    - RGB and flow:  ResNet3D from the paper 'A closer look at spatiotemporal convolutions for action recognition'. 
                    This is the r2plus1d in the mmaction packet
    - audio: ResNet: Deep residual learning for image recognition. In CVPR, 2016. 
    both with 50 layers. 

    - For fusion, we use a two-FC-layer network on concatenated features from visual and audio backbones, 
    followed by one prediction layer.
"""

import os

import numpy

import torch
import torch.nn as nn

from mmaction.models import backbones
from mmaction.models import heads
from mmaction.models import losses

from mmaction.models import build_model

from plato.models.multimodal import build_fc_from_config


class MM3F(nn.Module):
    """MMF network.
        This network supports the learning of several modalities

    Args:
        multimoda_model_configs (namedtuple): a namedtuple contains the configurations for
                                            different modalities, 'rgb_model', 'audio_model',
                                            'flow_model', 'text_model'
    """
    def __init__(
        self,
        multimoda_model_configs,  # multimodal_data_model
        separated_heads=True,  # whether each modality has its own prediction head
        fused_head=True
    ):  # a cls head makes prediction based on the fused multimodal feature
        super().__init__()

        if separated_heads:  # the model of each modality should contain its own head
            assert "cls_head" in list(multimoda_model_configs.rgb_model.keys())
            assert "cls_head" in list(
                multimoda_model_configs.flow_model.keys())
            assert "cls_head" in list(
                multimoda_model_configs.audio_model.keys())

        else:  # they share the same head to make classification
            assert fused_head == True  # they must have a shared head to make the classification

        self.rgb_model = build_model(multimoda_model_configs.rgb_model)
        self.flow_model = build_model(multimoda_model_configs.flow_model)
        self.audio_model = build_model(multimoda_model_configs.audio_model)

        if fused_head:
            fuse_model_config = multimoda_model_configs.fuse_model
            self.fuse_model = build_fc_from_config(fuse_model_config)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=self.init_std)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward_train(self, rgb_imgs, flow_imgs, audio_features, labels,
                      **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.rgb_model.with_cls_head and self.flow_model.with_cls_head and self.audio_model.with_cls_head

        rgb_imgs = rgb_imgs.reshape((-1, ) + rgb_imgs.shape[2:])
        flow_imgs = flow_imgs.reshape((-1, ) + flow_imgs.shape[2:])
        audio_features = audio_features.reshape((-1, ) +
                                                audio_features.shape[2:])
        losses = dict(rgb_losses=dict(),
                      flow_losses=dict(),
                      audio_losses=dict())

        # 1. forward the backbone
        rgb_feat = self.rgb_model.extract_feat(rgb_imgs)
        flow_feat = self.flow_model.extract_feat(flow_imgs)
        audio_feat = self.audio_model.extract_feat(audio_features)

        if fused_head:
            # obtain the fused feats
            fused_feat = torch.cat((rgb_feat, flow_feat, audio_feat), 1)
            fuse_cls_score = self.fuse_model(fused_feat)

        # 2. obtain the losses
        rgb_cls_score = self.rgb_model.cls_head(rgb_feat)
        flow_cls_score = self.flow_model.cls_head(flow_feat)
        audio_cls_score = self.audio_model.cls_head(audio_feat)

        gt_labels = labels.squeeze()
        rgb_loss_cls = self.rgb_model.cls_head.loss(rgb_cls_score, gt_labels,
                                                    **kwargs)
        flow_loss_cls = self.flow_model.cls_head.loss(flow_cls_score,
                                                      gt_labels, **kwargs)
        audio_loss_cls = self.audio_model.cls_head.loss(
            audio_cls_score, gt_labels, **kwargs)

        if fused_head:
            # obtain the fused feats
            fused_feat = torch.cat((rgb_feat, flow_feat, audio_feat), 1)
            fused_cls_score = self.fuse_model(fused_feat)
            fused_loss_cls = self.fuse_model.cls_head.loss(
                fused_cls_score, gt_labels, **kwargs)
            losses['fused_losses'].update(loss_cls)

        losses['rgb_losses'].update(loss_cls)
        losses['flow_losses'].update(flow_loss_cls)
        losses['audio_losses'].update(audio_loss_cls)

        return losses

    def forward(self,
                rgb_imgs,
                flow_imgs,
                audio_features,
                label=None,
                return_loss=True,
                **kwargs):
        """Defines the computation performed at every call.

        Args:
            rgb_imgs (torch.Tensor): The input data.
                the size of x is (num_batches, 3, 16, 112, 112).
            flow_imgs (torch.Tensor): The input data.
                the size of x is (num_batches, 3, 16, 112, 112).
            audio_features (torch.Tensor): The input data.
                the size of x is (num_batches, 3, 16, 112, 112).
        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            if self.rgb_model.blending is not None:
                rgb_imgs, label = self.rgb_model.blending(rgb_imgs, label)
            if self.flow_model.blending is not None:
                flow_imgs, label = self.flow_model.blending(flow_imgs, label)
            if self.audio_model.blending is not None:
                audio_features, label = self.audio_model.blending(
                    audio_features, label)

            return self.forward_train(rgb_imgs, flow_imgs, audio_features,
                                      label)

        return self.forward_test(rgb_imgs, flow_imgs, audio_features, **kwargs)