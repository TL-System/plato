"""
The implementation of the losses for the Calibre approach.

It includes NTXentLoss as the main loss, while:
    - prototype-oriented contrastive regularizer
    - prototype-based meta regularizer
    as optional auxiliary losses.

The one related loss is proposed in the paper:
Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf. However,
it belongs to one specific case, i.e., the fully-supervised, of Calibre.
"""

from typing import List
from collections import OrderedDict

import torch
import torch.nn as nn
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
