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

    - For fusion, we use a two-FC-layer network on concatenated features from
        visual and audio backbones,
    followed by one prediction layer.
"""

import logging

import torch.nn as nn

from plato.models.multimodal import base_net
from plato.models.multimodal import fusion_net


class DynamicMultimodalModule(nn.Module):
    """DynamicMultimodalModule network.
        This network supports the learning of several modalities (the modalities can be dynamic)

    Args:
        multimodal_nets_configs (namedtuple): a namedtuple contains the configurations for
                                            different modalities, 'rgb_model', 'audio_model',
                                            'flow_model', 'text_model'
    """
    def __init__(
        self,
        support_modality_names,
        multimodal_nets_configs,  # multimodal_data_model
        is_fused_head=True
    ):  # a cls head makes prediction based on the fused multimodal feature
        super().__init__()

        # ['rgb', "flow", "audio"]
        self.support_modality_names = support_modality_names
        self.support_nets = [
            mod_nm + "_model" for mod_nm in self.support_modality_names
        ]

        self.is_fused_head = is_fused_head

        assert all([
            s_net in multimodal_nets_configs.keys()
            for s_net in self.support_nets
        ])
        self.name_net_mapper = {}
        self.modality_fea_dims_mapper = {}
        for idx, modality_net in enumerate(self.support_nets):
            modality_name = self.support_modality_names[idx]
            modality_net = self.support_nets[idx]
            if modality_net in multimodal_nets_configs.keys():
                logging.info("Building the %s......", modality_net)
                net_config = multimodal_nets_configs[modality_net]
                is_head_included = "cls_head" in net_config.keys()
                logging.info("The head is defined")

                if is_head_included:
                    # the feature dimension is the input dimension of the cls head
                    fea_dims = net_config["cls_head"]["in_channels"]
                    self.modality_fea_dims_mapper[modality_name] = fea_dims

                self.name_net_mapper[
                    modality_name] = base_net.BaseClassificationNet(
                        net_configs=net_config,
                        is_head_included=is_head_included)

        if is_fused_head:

            fuse_net_config = multimodal_nets_configs["fuse_model"]

            if "modalities_feature_dim" in list(fuse_net_config.keys()):
                self.modality_fea_dims_mapper.update(
                    fuse_net_config["modalities_feature_dim"])

            self.cat_fusion_net = fusion_net.ConcatFusionNet(
                support_modalities=support_modality_names,
                modalities_fea_dim=self.modality_fea_dims_mapper,
                net_configs=fuse_net_config)

    def assing_weights(self, net_name, weights):
        """ Assign the weights to the specific network """
        self.name_net_mapper[net_name].load_state_dict(weights, strict=True)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            layer_module = getattr(self, f'layer{i}')
            layer_module.eval()
            for param in layer_module.parameters():
                param.requires_grad = False

    def forward(self, data_container, label=None, return_loss=True, **kwargs):
        """[Forward the data to the whole net]

        Args:
            data_container (dict): [key is the name of the modality
                                    while the value is batch of data]
            label (torch.tensor, optional): [the lable of the sample]. Defaults to None.
            return_loss (bool, optional): [whether return the loss ]. Defaults to True.
        """
        modalities_pred_scores_container = dict()
        modalities_losses_container = dict()
        modalities_features_container = dict()

        for modality_name in data_container.keys():

            modality_net = self.name_net_mapper[modality_name]
            modality_ipt_data = data_container[modality_name]
            batch_size = modality_ipt_data.shape[0]

            logging.debug("modality_name: %s", modality_name)
            logging.debug("modality_net: %s", modality_net)
            logging.debug("modality_net inner net: %s", modality_net.get_net())
            logging.debug("modality_ipt_data: %s", modality_ipt_data.shape)
            logging.debug("batch_size: %s", batch_size)

            # obtain the modality fea and the class opt
            modality_opt = modality_net.forward(ipt_data=modality_ipt_data,
                                                label=label,
                                                return_loss=return_loss)

            modalities_features_container[modality_name] = modality_opt[0]
            modalities_pred_scores_container[modality_name] = modality_opt[1]
            modalities_losses_container[modality_name] = modality_opt[2]

        if self.is_fused_head:
            # obtain the fused feats by concating the modalities features
            #   The order should follow the that in the support_modality_names
            fused_feat = self.cat_fusion_net.create_fusion_feature(
                batch_size=batch_size,
                modalities_features_container=modalities_features_container)
            fused_cls_score, fused_loss = self.cat_fusion_net.forward(
                fused_feat, label, return_loss=return_loss)
            modalities_pred_scores_container["fused"] = fused_cls_score
            modalities_losses_container["fused"] = fused_loss

        return modalities_pred_scores_container, modalities_losses_container
