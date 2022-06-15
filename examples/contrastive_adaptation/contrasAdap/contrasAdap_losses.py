"""
The implementation of the losses for our contrastive adaptation.

"""

import logging

import torch
import torch.nn as nn


class ContrasAdapLoss(nn.Module):
    """
    The contrastive adaptation losses for our proposed method.
    It mainly includes:
        - the client-oriented Representation Disentanglement loss
        - the adversarial alignment loss

    """

    def __init__(self,
                 temperature=0.07,
                 contrast_mode='all',
                 base_temperature=0.07,
                 device=None):
        super().__init__()
        # a hyper-parameter constant
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def client_specific_representation_loss(self,
                                            features,
                                            labels=None,
                                            mask=None):
        """Compute loss to support the a client specific representation.
        The major contribution is that: the representation learned by the
        contrastive ssl with similarity/dissimilarity measurements

        A. potentially forgets the distribution information of the client,
        such as the local samples' distance of inter-classes and intra-classes.

        B. is prone to over-learn the common representation for participating
        clients by extracting high-level principles from raw local samples but losts
        the ability of generalizing well to downstream tasks.

        The insight A motivates the introduce the loss containing clusters'
        information, i.e., supervision information, in a contrastive manner.

        The insight B motivates the meta-training procedure to be included
        in the ssl training stage in each client. This is to say the representation
        should be further used to complete the cliet-specific task based on the local
        dataset.

        The work most related to our idea of motivation A is the
        Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf. However,
        it belongs to one specific case, i.e., the fully-supervised, of our method.

        If both `labels` and `mask` are None, it degenerates to SimCLR's
        unsupervised contrastive loss.

        Args:
            features (list): the obtaind features. Each item is a tensor with
                            shape batch_size, feature_dim.
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        batch_size = features[0].shape[0]
        device = features[0].get_device()
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError(
                    'Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # obtain the number of cotrastive items
        contrast_count = len(contrast_feature)
        # obtain the combined features
        # batch_size * contrast_count, feature_dim
        contrast_feature = torch.cat(contrast_feature, dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
