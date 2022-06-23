"""
The implementation of the losses for our contrastive adaptation.


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

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from plato.utils import ssl_losses


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
                 batch_size=128):
        super().__init__()
        # a hyper-parameter constant
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.batch_size = batch_size

        self.cross_sg_criterion = ssl_losses.CrossStopGradientL2loss()

        self.ntx_criterion = ssl_losses.NTXent(self.batch_size,
                                               self.temperature,
                                               world_size=1)

    def clients_invariant_representation_similarity_loss(self, outputs):
        """ Motivate the similarity of predicted features and projected features.

            - Can be the loss function used in the BYOL method.

            The losses in this case only denote similarity between one sample
            and its augmented sample.

            The outputs should contain two tuples of features
            (online_pred_one, online_pred_two)
            (target_proj_one, target_proj_two)

        """

        return self.cross_sg_criterion(outputs)

    def clients_invariant_representation_loss(self, encoded_z1, encoded_z2):
        """ Motivate the representation to be clients invariant by using the
            within one batch:
            1. data argument of one sample as its positive sample.
            2. All other samples as the negative samples.

            - Can be the loss function used in the SimCLR method.

            The losses in this case do not introudce the supervision informaton
            such as the labels but directly utilize the samples within one batch
            to generate positive and negative samples.
            For one sample,
                the positive sample is its augmented sample obtained in the transform
                the negative samples are all other samples within one batch.
        """
        ntx_loss = self.ntx_criterion(encoded_z1, encoded_z2)

        return ntx_loss

    def client_semi_invariant_prototype_representation_loss(
            self, encoded_z1, encoded_z2, labels):
        """ Compute loss to support the representation that is semi-invariant
            among clients.

            The name of the 'semi' is because we utilize the client's labels to
            build the prototype within one batch.
        """

        pass

    def client_specific_representation_loss(self, encoded_z1, encoded_z2,
                                            labels):
        """ Compute loss to support the a client specific representation.

            Takes `features` and `labels` as input, and return the loss.

            If both `labels` and `mask` are None, it degenerates to SimCLR's
            unsupervised contrastive loss.

            Args:
                encoded_z1 (torch.tensor): the obtaind features. batch_size, feature_dim.
                encoded_z2 (torch.tensor): the obtaind features. batch_size, feature_dim.
                labels: ground truth of shape [bsz].

            Returns:
                A loss scalar.

        """
        # Combine two inputs features into one
        # encoded_z1: batch_size, fea_dim
        # encoded_z2: batch_size, fea_dim
        # features: batch_size, 2, fea_dim

        features = torch.cat(
            [encoded_z1.unsqueeze(1),
             encoded_z2.unsqueeze(1)], dim=1)

        features = F.normalize(features, dim=-1, p=2)
        batch_size = features[0].shape[0]
        device = features[0].get_device()

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        # mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
        #     has the same class as sample i. Can be asymmetric.
        mask = torch.eq(labels, labels.T).float().to(device)

        # obtain the number of cotrastive items
        contrast_count = features.shape[1]
        # obtain the combined features
        # batch_size * contrast_count, feature_dim
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

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

    def meta_learning_loss(self, logits, labels):
        """ The meta learning is the performance of the pre-trained ssl method
            on the local classification task. I.e., we directly utilize the
            cross entropy loss. """

        return torch.nn.CrossEntropyLoss(logits, labels)