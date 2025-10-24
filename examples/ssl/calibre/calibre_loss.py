"""
The implementation of the losses for the Calibre approach.

The objective function of Calibre is the combination of the main loss
and the auxiliary loss containing two regularizers. And, Calibre assigns
different weights to these two parts. We build the loss function to be
general enough so that which loss will be included in the objective function
and the corresponding weight can be set under the `algorithm` block of the config file.

The main loss will be a NTXentLoss while the two regularizers are:
    - prototype-oriented contrastive regularizer
    - prototype-based meta regularizer

The objective function related to ours is proposed by Supervised Contrastive Learning
(SupCon) with the paper address https://arxiv.org/pdf/2004.11362.pdf.
However, this supervised objective function lags behind our unsupervised objective
function. Besides, it does not have theoretically obtained regularizers and overall
structure but only computes clustering loss with ground truth labels.
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from clustering import kmeans_clustering
from lightly import loss as lightly_loss
from prototype_loss import get_prototype_loss
from torch import nn

from plato.trainers import loss_criterion


def _to_dict(source: Any) -> dict[str, Any]:
    """Convert configuration-like objects into plain dictionaries."""
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return {str(key): value for key, value in source.items()}
    if hasattr(source, "_asdict"):
        as_dict = source._asdict()
        if isinstance(as_dict, Mapping):
            return {str(key): value for key, value in as_dict.items()}
    raise TypeError(f"Unsupported configuration type: {type(source)!r}")


class CalibreLoss(nn.Module):
    """
    The contrastive adaptation losses for Calibre.
    """

    def __init__(
        self,
        main_loss: str,
        main_loss_params: Mapping[str, Any] | None,
        auxiliary_losses: Sequence[str] | None = None,
        auxiliary_loss_params: Mapping[str, Any] | None = None,
        losses_weight: Mapping[str, Any] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()

        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_main_loss", main_loss)

        main_loss_parameters = (
            dict(main_loss_params) if main_loss_params is not None else {}
        )
        object.__setattr__(self, "_main_loss_params", main_loss_parameters)

        auxiliary_losses_list = list(auxiliary_losses or [])

        try:
            auxiliary_loss_params_dict = _to_dict(auxiliary_loss_params)
        except TypeError:
            auxiliary_loss_params_dict = {}
        normalized_aux_params: dict[str, dict[str, Any]] = {}
        for key, value in auxiliary_loss_params_dict.items():
            if isinstance(value, Mapping):
                normalized_aux_params[key] = dict(value)
            elif hasattr(value, "_asdict"):
                normalized_aux_params[key] = _to_dict(value)
            else:
                normalized_aux_params[key] = {}

        try:
            losses_weight_dict = _to_dict(losses_weight)
        except TypeError:
            losses_weight_dict = {}

        self.loss_weights_params: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.loss_functions: OrderedDict[str, Any] = OrderedDict()

        weight = losses_weight_dict.get(main_loss, 1.0)
        self.loss_weights_params[main_loss] = {
            "params": main_loss_parameters,
            "weight": weight,
        }

        for loss in auxiliary_losses_list:
            if loss not in losses_weight_dict:
                continue
            params_for_loss = normalized_aux_params.get(loss, {})
            if hasattr(params_for_loss, "_asdict"):
                params_for_loss = _to_dict(params_for_loss)
            self.loss_weights_params[loss] = {
                "params": params_for_loss,
                "weight": losses_weight_dict[loss],
            }

        # Align the loss name with its corresponding function
        for loss_name in self.loss_weights_params:
            if hasattr(self, loss_name):
                loss_fn = getattr(self, loss_name)
            else:
                loss_fn = loss_criterion.get(
                    loss_criterion=loss_name,
                    loss_criterion_params=self.loss_weights_params[loss_name]["params"],
                )

            self.loss_functions[loss_name] = loss_fn

    def prototype_regularizers(self, encodings, projections, **kwargs):
        """Compute the L_p and L_n losses mentioned the paper."""
        n_clusters = kwargs["n_clusters"]
        distance_type = kwargs["distance_type"]
        # Get encodings, each with shape:
        # [batch_size, feature_dim]
        encodings_a, encodings_b = encodings
        # Get projections, each with shape:
        # [batch_size, projection_dim]
        projections_a, projections_b = projections

        batch_size = encodings_a.shape[0]

        # Perform the K-means clustering to get the prototypes
        # shape: [2*batch_size, feature_dim]
        full_encodings = torch.cat((encodings_a, encodings_b), dim=0)
        # Get cluster assignment for the input encodings,
        # clusters_assignment shape, [2*batch_size]
        clusters_assignment, _ = kmeans_clustering(
            full_encodings, n_clusters=n_clusters
        )
        # Get the unique cluster ids
        # with shape, [n_clusters]
        pseudo_classes = torch.unique(clusters_assignment)

        # Split into two parts corresponding to a, and b
        # each with shape, [batch_size]
        pseudo_labels_a, pseudo_labels_b = torch.split(clusters_assignment, batch_size)

        ## prototype-oriented contrastive regularizer
        # Compute the prototype features based on projection
        # Filter out clusters that don't have samples in both views to avoid NaN
        valid_classes = []
        prototypes_a_list = []
        prototypes_b_list = []
        support_prototypes_list = []

        for class_id in pseudo_classes:
            mask_a = pseudo_labels_a == class_id
            mask_b = pseudo_labels_b == class_id

            # Only include clusters that have samples in both views
            if mask_a.sum() > 0 and mask_b.sum() > 0:
                valid_classes.append(class_id)
                prototypes_a_list.append(projections_a[mask_a].mean(0))
                prototypes_b_list.append(projections_b[mask_b].mean(0))
                support_prototypes_list.append(encodings_a[mask_a].mean(0))

        # Stack the prototypes
        # with shape, [n_valid_clusters, projection_dim]
        prototypes_a = torch.stack(prototypes_a_list, dim=0)
        # With shape, [n_valid_clusters, projection_dim]
        prototypes_b = torch.stack(prototypes_b_list, dim=0)

        # Compute the L_p loss
        loss_fn = lightly_loss.NTXentLoss(memory_bank_size=0)
        l_p = loss_fn(prototypes_a, prototypes_b)

        # Compute prototype-based meta regularizer
        # Support set with shape, [n_valid_clusters, encoding_dim]
        support_prototypes = torch.stack(support_prototypes_list, dim=0)

        # Remap pseudo_labels_b to new indices (0 to n_valid_clusters-1)
        valid_classes_tensor = torch.tensor(
            valid_classes, device=pseudo_labels_b.device
        )
        # Create a mapping from old class IDs to new indices
        remapped_labels = torch.zeros_like(pseudo_labels_b)
        for new_idx, class_id in enumerate(valid_classes):
            remapped_labels[pseudo_labels_b == class_id] = new_idx

        # Filter queries to only include samples from valid clusters
        valid_query_mask = torch.isin(pseudo_labels_b, valid_classes_tensor)
        queries_filtered = encodings_b[valid_query_mask]
        labels_filtered = remapped_labels[valid_query_mask]

        # Calculate distances between query set embeddings and class prototypes
        l_n = get_prototype_loss(
            support_prototypes,
            queries=queries_filtered,
            query_labels=labels_filtered,
            distance_type=distance_type,
        )

        return l_p, l_n

    def forward(self, *args, **kwargs):
        """Forward the loss computaton layer."""
        total_loss = 0.0
        # Extract terms from the input
        encodings = args[0]
        projections = args[1]

        # Visit the loss container to compute the whole loss
        # in which each term is loss_weight * loss
        for loss_name in self.loss_weights_params:
            loss_weight = self.loss_weights_params[loss_name]["weight"]
            loss_params = self.loss_weights_params[loss_name]["params"]

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
                computed_loss = self.loss_functions[loss_name](*projections)

                total_loss += loss_weight * computed_loss

        return total_loss
