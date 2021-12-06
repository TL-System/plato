# @Date    : 2021-06-27 13:23:05
"""
This multimodal network is the core network used in our paper.
    It can receives three datasets from three modalities(RGB, optical flow, and audio)
    and then process them with
three different networks:
    - RGB and flow:  ResNet3D from the paper 'A closer look at spatiotemporal
                    convolutions for action recognition'.
                    This is the r2plus1d in the mmaction packet
    - audio: ResNet: Deep residual learning for image recognition. In CVPR, 2016.
    both with 50 layers.

    - For fusion, we use a two-FC-layer network on concatenated
        features from visual and audio backbones,
    followed by one prediction layer.
"""

import torch
import torch.nn as nn

from mmaction.models import build_loss

from plato.models.multimodal import fc_net


class ConcatFusionNet(nn.Module):
    """ This supports concat the features of different modalities to one vector"""
    def __init__(self, support_modalities, modalities_fea_dim, net_configs):
        super(ConcatFusionNet, self).__init__()

        # the support modality name is the pre-defined order that must be
        # followed in the forward process
        #   especially in the fusion part
        self.support_modality_names = support_modalities  # a list
        self.modalities_fea_dim = modalities_fea_dim
        self.net_configs = net_configs
        # 1 build the model based on the configurations
        self._fuse_net = fc_net.build_fc_from_config(net_configs)

        self.loss_cls = build_loss(self.net_configs["loss_cls"])

    def create_fusion_feature(self, batch_size, modalities_features_container):
        """[summary]

        Args:
            modalities_features_container (dict): [key is the name of the modality
                                                while the value is the corresponding features]
            modalities_features_dims_container (dict): [key is the name of the modality
                                                while the value is the defined dim of the feature]
        """
        # obtain the fused feats by concating the modalities features
        #   The order should follow the that in the support_modality_names
        modalities_feature = []
        for modality_name in self.support_modality_names:
            if modality_name not in modalities_features_container:
                modality_dim = self.modalities_fea_dim[modality_name]
                # insert all zeros features if that modality is missing
                modality_feature = torch.zeros(size=(batch_size, modality_dim))
            else:
                modality_feature = modalities_features_container[modality_name]

            modalities_feature.append(modality_feature)

        fused_feat = torch.cat(modalities_feature, 1)

        return fused_feat

    def forward(self, fused_features, gt_labels, return_loss):
        """ Forward the network """
        fused_cls_score = self._fuse_net(fused_features)

        if return_loss:
            fused_loss = self.loss_cls(fused_cls_score, gt_labels)

            return [fused_cls_score, fused_loss]
        else:
            return [fused_cls_score]
