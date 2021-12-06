"""
The base class used for all following classes

"""
import torch
import torch.nn as nn

from mmaction.models import build_model


class BaseClassificationNet(nn.Module):
    """ Base class for classification networks """
    def __init__(self, net_configs, is_head_included=True):
        super(BaseClassificationNet, self).__init__()

        self.net_configs = net_configs
        # 1 build the model based on the configurations
        self._net = build_model(net_configs)

        self.is_head_included = is_head_included

        # the features must be forwarded the avg pool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def get_net(self):
        """ Get the built network """
        return self._net

    def forward_train(self, ipt_data, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        ipt_data_sz = ipt_data.reshape((-1, ) + ipt_data.shape[2:])

        # 1. forward the backbone
        data_feat = self._net.extract_feat(ipt_data_sz)
        # from [N * num_segs, in_channels, h, w]
        #   to [N, in_channels, 1, 1]
        immediate_feat = self.avg_pool(data_feat)
        # to [N, in_channels]
        immediate_feat = torch.squeeze(immediate_feat)

        # 2. forward the classification head if possible and obtain the losses
        loss_cls = 0.0
        if self.is_head_included:
            cls_score = self._net.cls_head(data_feat)

            gt_labels = labels.squeeze()
            loss_cls = self._net.cls_head.loss(cls_score, gt_labels, **kwargs)

            return [immediate_feat, cls_score, loss_cls]

        return [immediate_feat]

    def forward_test(self, ipt_data, **kwargs):
        """Defines the computation performed at every call when training."""

        ipt_data = ipt_data.reshape((-1, ) + ipt_data.shape[2:])
        # 1. forward the backbone
        data_feat = self._net.extract_feat(ipt_data)
        # 2. forward the classification head if possible and obtain the losses
        cls_score = 0.0
        if self.is_head_included:
            cls_score = self._net.cls_head(data_feat)

            return [data_feat, cls_score]

        return [data_feat]

    def forward(self, ipt_data, label=None, return_loss=True, **kwargs):
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
            else:
                blended_ipt_data = ipt_data
            return self.forward_train(blended_ipt_data, label, **kwargs)

        return self.forward_test(ipt_data, **kwargs)
