"""
The implementation of the losses for the Calibre approach.

It includes NTXentLoss as the main loss, while:
    - prototype-oriented contrastive regularizer
    - prototype-based meta regularizer
    - prototype_contrastive_representation_loss
    - meta_prototype_contrastive_representation_loss
    as optional auxiliary losses.

The one related loss is proposed in the paper:
Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf. However,
it belongs to one specific case, i.e., the fully-supervised, of Calibre.
"""

from typing import List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly import loss as lightly_loss

from plato.trainers import loss_criterion

from clustering import kmeans_clustering
from prototype_loss import get_prototype_loss


class CalibreLoss(nn.Module):
    """
    The contrastive adaptation losses for Calibre.
    """

    def __init__(
        self,
        main_loss: str,
        main_loss_params: dict,
        auxiliary_losses: List[str] = None,
        auxiliary_losses_params: List[dict] = None,
        losses_weight: List[float] = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device

        self.main_loss = main_loss
        self.main_loss_params = main_loss_params

        self.auxiliary_losses = auxiliary_losses
        self.auxiliary_losses_params = auxiliary_losses_params

        self.losses_weight = losses_weight._asdict()

        self.losses = OrderedDict()
        self.losses_func = OrderedDict()
        self.set_default()
        self.set_losses()

        self.define_losses_func()

    def set_default(self):
        """Setting the default terms."""

        self.auxiliary_losses = (
            self.auxiliary_losses if self.auxiliary_losses is not None else []
        )
        self.auxiliary_losses_params = (
            self.auxiliary_losses_params
            if self.auxiliary_losses_params is not None
            else []
        )
        assert len(self.auxiliary_losses) == len(self.auxiliary_losses_params)

        self.losses[self.main_loss] = {"params": {}, "weight": 0.0}
        if self.main_loss not in self.losses_weight:
            self.losses_weight[self.main_loss] = 1.0

        for name in self.auxiliary_losses:
            if name not in self.losses_weight:
                self.losses_weight[name] = 0.0

            self.losses[name] = {"params": {}, "weight": 0.0}

    def set_losses(self):
        """Setting the losses and the corresponding parameters."""
        self.losses[self.main_loss]["params"] = self.main_loss_params
        self.losses[self.main_loss]["weight"] = self.losses_weight[self.main_loss]

        for loss in self.auxiliary_losses:
            param = self.auxiliary_losses_params[loss]
            self.losses[loss]["params"] = param._asdict()
            self.losses[loss]["weight"] = self.losses_weight[loss]

    def define_losses_func(self):
        """Define the loss functions."""

        for loss_name in self.losses:
            if hasattr(self, loss_name):
                loss_func = getattr(self, loss_name)
            else:
                loss_func = loss_criterion.get(
                    loss_criterion=loss_name,
                    loss_criterion_params=self.losses[loss_name]["params"],
                )

            self.losses_func[loss_name] = loss_func

    def prototype_regularizers(self, encodings, projections, **kwargs):
        """Compute the L_p and L_n losses mentioned the paper."""
        n_clusters = kwargs["n_clusters"]
        distance_type = kwargs["distance_type"]
        # get encodings, each with shape:
        # [batch_size, feature_dim]
        encodings_a, encodings_b = encodings
        # get projections, each with shape:
        # [batch_size, projection_dim]
        projections_a, projections_b = projections

        batch_size = encodings_a.shape[0]

        # perform the K-means clustering to get the prototypes
        # shape: [2*batch_size, feature_dim]
        full_encodings = torch.cat((encodings_a, encodings_b), axis=0)
        # get cluster assignment for the input encodings,
        # clusters_assignment shape, [2*batch_size]
        clusters_assignment, _ = kmeans_clustering(
            full_encodings, n_clusters=n_clusters, device=self.device
        )
        # get the unique cluster ids
        # with shape, [n_clusters]
        pseudo_classes = torch.unique(clusters_assignment)
        # split into two parts corresponding to a, and b
        # each with shape, [batch_size]

        pseudo_labels_a, pseudo_labels_b = torch.split(clusters_assignment, batch_size)

        ##
        ## prototype-oriented contrastive regularizer
        ##
        # compute the prototype features based on projection
        # with shape, [n_clusters, projection_dim]
        prototypes_a = torch.stack(
            [
                projections_a[pseudo_labels_a == class_id].mean(0)
                for class_id in pseudo_classes
            ],
            dim=0,
        )
        # with shape, [n_clusters, projection_dim]
        prototypes_b = torch.stack(
            [
                projections_b[pseudo_labels_b == class_id].mean(0)
                for class_id in pseudo_classes
            ],
            dim=0,
        )

        # Compute the L_p loss
        loss_fn = lightly_loss.NTXentLoss(memory_bank_size=0)
        l_p = loss_fn(prototypes_a, prototypes_b)
        ##
        ## Compute prototype-based meta regularizer
        ##
        # support set
        # with shape, [n_clusters, encoding_dim]
        support_prototypes = torch.stack(
            [
                encodings_a[pseudo_labels_a == class_id].mean(0)
                for class_id in pseudo_classes
            ],
            dim=0,
        )

        # Calculate distances between query set embeddings and class prototypes
        # with shape, [n_clusters, encoding_dim]
        l_n = get_prototype_loss(
            support_prototypes,
            queries=encodings_b,
            query_labels=pseudo_labels_b,
            distance_type=distance_type,
        )

        return l_p, l_n

    def prototype_contrastive_representation_loss(self, outputs, labels):
        """Compute loss to support the a client specific representation.

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
        batch_size = encoded_z1.shape[0]

        device = encoded_z1.get_device() if encoded_z1.is_cuda else torch.device("cpu")

        encoded_z1 = F.normalize(encoded_z1, dim=1)
        encoded_z2 = F.normalize(encoded_z2, dim=1)

        # batch_size, 2, dim
        features = torch.cat([encoded_z1.unsqueeze(1), encoded_z2.unsqueeze(1)], dim=1)

        features = features.view(features.shape[0], features.shape[1], -1)

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        # mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
        #     has the same class as sample i. Can be asymmetric.
        mask = torch.eq(labels, labels.T).float().to(device)

        # obtain the number of cotrastive items
        contrast_count = features.shape[1]
        # obtain the combined features
        # batch_size * contrast_count, feature_dim
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
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

    def forward(self, *args, **kwargs):
        """Forward the loss computaton layer."""
        total_loss = 0.0
        labels = kwargs.get("labels", None)
        encodings = args[0]
        projections = args[1]

        for loss_name in self.losses:
            loss_weight = self.losses[loss_name]["weight"]
            loss_params = self.losses[loss_name]["params"]

            if loss_name == "prototype_regularizers":
                regularizers_loss = self.prototype_regularizers(
                    encodings=encodings, projections=projections, **loss_params
                )

                computed_loss = sum(
                    loss * loss_weight[loss_idx]
                    for loss_idx, loss in enumerate(regularizers_loss)
                )
                total_loss += computed_loss
            else:
                computed_loss = self.losses_func[loss_name](*projections)

                total_loss += loss_weight * computed_loss

        return total_loss
