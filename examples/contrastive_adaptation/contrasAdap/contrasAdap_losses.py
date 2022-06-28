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

    def __init__(
        self,
        losses,
        supervision_contrastive_lambda=1.0,
        similarity_lambda=0.0,
        ntx_lambda=0.0,
        meta_lambda=0.0,
        meta_contrastive_lambda=0.0,
        perform_label_distortion=False,
        label_distrotion_type="random",
        temperature=0.07,
        base_temperature=0.07,
        contrast_mode='all',
    ):
        super().__init__()

        # the losses to be computed
        # the enabled losses can be:
        # * sup_contrastive_loss  - client_specific_representation_loss
        # * sim_contrastive_loss  - clients_invariant_representation_similarity_loss
        # * ntx_contrastive_loss  - clients_invariant_representation_loss
        # * prot_repre_loss       - prototype_representation_loss
        # * prot_contrastive_loss - prototype_contrastive_representation_loss
        self.losses = losses

        # a hyper-parameter for the supervision_contrastive loss
        self.supervision_contrastive_lambda = supervision_contrastive_lambda
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        # weight for the similarity contrastive loss
        self.similarity_lambda = similarity_lambda

        # weight for the ntx contrastive loss
        self.ntx_lambda = ntx_lambda

        # weight for the prototype losses
        self.meta_lambda = meta_lambda

        self.meta_contrastive_lambda = meta_contrastive_lambda

        # whether to perform the label distortion
        self.perform_label_distortion = perform_label_distortion
        self.label_distrotion_type = label_distrotion_type

        self.losses_func = {
            "sup_contrastive_loss": self.client_specific_representation_loss,
            "sim_contrastive_loss":
            self.clients_invariant_representation_similarity_loss,
            "ntx_contrastive_loss": self.clients_invariant_representation_loss,
            "prot_repre_loss": self.prototype_representation_loss,
            "prot_contrastive_loss":
            self.prototype_contrastive_representation_loss
        }

        self.losses_weight = {
            "sup_contrastive_loss": self.supervision_contrastive_lambda,
            "sim_contrastive_loss": self.similarity_lambda,
            "ntx_contrastive_loss": self.ntx_lambda,
            "prot_repre_loss": self.meta_lambda,
            "prot_contrastive_loss": self.meta_contrastive_lambda
        }

    def clients_invariant_representation_similarity_loss(
            self, outputs, labels=None):
        """ Motivate the similarity of predicted features and projected features.

            - Can be the loss function used in the BYOL method.

            The losses in this case only denote similarity between one sample
            and its augmented sample.

            The outputs should contain two tuples of features
            (online_pred_one, online_pred_two)
            (target_proj_one, target_proj_two)

        """

        cross_sg_criterion = ssl_losses.CrossStopGradientL2loss()

        return cross_sg_criterion(outputs)

    def clients_invariant_representation_loss(self, outputs, labels=None):
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
        encoded_z1, encoded_z2 = outputs

        self.ntx_criterion = ssl_losses.NTXent(self.temperature, world_size=1)
        ntx_loss = self.ntx_criterion(encoded_z1, encoded_z2)

        return ntx_loss

    def random_label_distortion(self, gt_labels):
        """ To cover the label information for better generalization,
            each client should perform the label distortion to shift the label
            index. """
        pseudo_labels = torch.zeros(gt_labels.shape)
        gt_classes = torch.unique(gt_labels).cpu().numpy()
        num_classes = len(num_classes)
        new_classes_label_mapper = {
            cls: cls_idx
            for cls_idx, cls in enumerate(gt_classes)
        }
        for cls in gt_classes:
            pseudo_labels[gt_classes == cls] = new_classes_label_mapper[cls]

        return pseudo_labels

    def prototype_representation_loss(self, outputs, labels):
        """ Compute loss to support the representation that is semi-invariant
            among clients.

            The name of the 'semi' is because we utilize the client's labels to
            build the prototype within one batch. By doing so, the training
            process and the learning process do not utilize the supervision
            information directly, making the learned representation generalize
            well to downstream tasks.

            To compute this loss, we first need to create:
                - support set, S_i - encoded_z1
                - query set, Q_i - encoded_z2
            Then, there are two stages:
                In the first stage, there are two schemas:
            Schema A:
                - the prototypes are generated based on the representation
                of the support set
                - a distribution over classes for the query set Q_i is
                produced based on a softmax over distances to the
                prototypes in the embedding space. The negative log-probability
                is shown as

            Schema B:

        """
        encoded_z1, encoded_z2 = outputs
        # Infer the number of different classes from the labels of the support set
        prototypes = torch.unique(labels)
        num_prototypes = len(prototypes)
        # Prototype i is the mean of all instances of features corresponding to labels == i
        encoded_prototypes = torch.cat([
            encoded_z1[torch.nonzero(labels == label)].mean(0)
            for label in range(num_prototypes)
        ])
        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(encoded_z2, encoded_prototypes)

        # And here is the super complicated operation to transform those distances into classification scores!
        meta_scores = -dists

        meta_loss = nn.CrossEntropyLoss(meta_scores, labels)

        return meta_loss

    def client_specific_representation_loss(self, outputs, labels):
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
        encoded_z1, encoded_z2 = outputs
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

    def prototype_contrastive_representation_loss(self, outputs, labels):
        """ Compute the contrastive loss based on the prototypes instead of
            each sample.

            The support set (encoded_z1) is utilized to create the prototypes
            in which each prototype is the averaged embedding of samples
            Then, the contrastive loss is computed based in the built prototypes.

             """
        encoded_z1, encoded_z2 = outputs

        # Infer the number of different classes from the labels of the support set
        prototypes_label = torch.unique(labels)
        num_prototypes = len(prototypes_label)

        # Prototype i is the mean of all instances of features corresponding to labels == i
        view1_prototypes = torch.cat([
            encoded_z1[torch.nonzero(
                labels == prototypes_label[label_idx])].mean(0)
            for label_idx in range(num_prototypes)
        ])
        view2_prototypes = torch.cat([
            encoded_z2[torch.nonzero(
                labels == prototypes_label[label_idx])].mean(0)
            for label_idx in range(num_prototypes)
        ])

        return self.client_specific_representation_loss(
            [view1_prototypes, view2_prototypes], prototypes_label)

    def forward(self, outputs, labels):
        """ Forward the loss computaton module. """
        total_loss = 0.0
        for loss_name in self.losses:
            loss_weight = self.losses_weight[loss_name]
            computed_loss = self.losses_func[loss_name](outputs, labels)

            total_loss += loss_weight * computed_loss

        return total_loss