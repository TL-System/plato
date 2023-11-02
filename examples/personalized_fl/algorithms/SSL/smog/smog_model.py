"""
A model for the SMoG method.
"""
import copy

import torch
from torch import nn
from sklearn.cluster import KMeans

from lightly.models.utils import deactivate_requires_grad
from lightly.models.modules.heads import (
    SMoGPredictionHead,
    SMoGProjectionHead,
    SMoGPrototypes,
)

from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config


class SMoG(nn.Module):
    """Core structure of the SMoG model."""

    def __init__(self, encoder=None):
        super().__init__()
        self.temperature = (
            Config().trainer.smog_temperature
            if hasattr(Config().trainer, "smog_temperature")
            else 0.1
        )

        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )
        # Define the encoder.
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )

        # Define the encoder based on the model_name in config.
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = encoder_registry.get(
                model_name=encoder_name, **encoder_params
            )

        self.encoding_dim = self.encoder.encoding_dim
        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim
        prediction_hidden_dim = Config().trainer.prediction_hidden_dim
        prediction_out_dim = Config().trainer.prediction_out_dim

        # Define the projector.
        self.projection_head = SMoGProjectionHead(
            self.encoding_dim, projection_hidden_dim, projection_out_dim
        )
        self.prediction_head = SMoGPredictionHead(
            projection_out_dim, prediction_hidden_dim, prediction_out_dim
        )

        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.n_groups = Config().trainer.n_groups
        n_prototypes = Config().trainer.n_prototypes
        beta = Config().trainer.smog_beta

        # Define the prototypes
        self.smog = SMoGPrototypes(
            group_features=torch.rand(self.n_groups, n_prototypes), beta=beta
        )

        # Current iteration.
        self.n_iteration = 0

    def _cluster_features(self, features: torch.Tensor) -> torch.Tensor:
        """Cluster the features using sklearn."""
        # Cluster the features using sklearn
        # (note: faiss is probably more efficient).
        features = features.cpu().numpy()
        kmeans = KMeans(self.n_groups).fit(features)
        clustered = torch.from_numpy(kmeans.cluster_centers_).float()
        clustered = torch.nn.functional.normalize(clustered, dim=1)
        return clustered

    def reset_group_features(self, memory_bank):
        """Reset the group features based on the clusters."""
        # See https://arxiv.org/pdf/2207.06167.pdf Table 7b).
        features = memory_bank.bank
        group_features = self._cluster_features(features.t())
        self.smog.set_group_features(group_features)

    def reset_momentum_weights(self):
        """Reset the momentum weights."""
        # See https://arxiv.org/pdf/2207.06167.pdf Table 7b).
        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward_view(self, views):
        """Forward views of the samples."""
        features = self.encoder(views).flatten(start_dim=1)
        encoded = self.projection_head(features)
        predicted = self.prediction_head(encoded)
        return encoded, predicted

    def forward_momentum(self, samples):
        """Forward the momentum mechanism."""
        features = self.encoder_momentum(samples).flatten(start_dim=1)
        encoded = self.projection_head_momentum(features)
        return encoded

    def forward(self, multiview_samples):
        """Forward the multiview samples."""
        views1, views2 = multiview_samples

        if self.n_iteration % 2:
            # Swap batches every two iterations.
            views2, views1 = views1, views2

        views1_encoded, views1_predicted = self.forward_view(views1)

        views2_encoded = self.forward_momentum(views2)

        # Update group features and get group assignments.
        assignments = self.smog.assign_groups(views2_encoded)
        group_features = self.smog.get_updated_group_features(views1_encoded)
        logits = self.smog(
            views1_predicted, group_features, temperature=self.temperature
        )
        self.smog.set_group_features(group_features)

        return logits, assignments
