"""
The model for the SMoG algorithm.
"""

import copy

import torch
from lightly.models.modules.heads import (
    SMoGPredictionHead,
    SMoGProjectionHead,
    SMoGPrototypes,
)
from lightly.models.utils import deactivate_requires_grad
from sklearn.cluster import KMeans
from torch import nn

from plato.config import Config
from plato.models.cnn_encoder import Model as encoder_registry


class SMoG(nn.Module):
    """The structure of the SMoG model."""

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

        # Define the projector.
        self.projector = SMoGProjectionHead(
            self.encoder.encoding_dim,
            Config().trainer.projection_hidden_dim,
            Config().trainer.projection_out_dim,
        )
        self.predictor = SMoGPredictionHead(
            Config().trainer.projection_out_dim,
            Config().trainer.prediction_hidden_dim,
            Config().trainer.prediction_out_dim,
        )

        # Deepcopy the encoder and projector to create the momentum
        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projector_momentum = copy.deepcopy(self.projector)

        # Deactivate the requires_grad flag for all parameters
        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projector_momentum)

        # Set the necessary hyper-parameter for SMoG
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
        features = features.cpu().numpy()
        kmeans = KMeans(self.n_groups).fit(features)
        clustered = torch.from_numpy(kmeans.cluster_centers_).float()
        clustered = torch.nn.functional.normalize(clustered, dim=1)
        return clustered

    def reset_group_features(self, memory_bank):
        """Reset the group features based on the clusters."""
        features = memory_bank.bank
        group_features = self._cluster_features(features.t())
        self.smog.set_group_features(group_features)

    def reset_momentum_weights(self):
        """Reset the momentum weights."""
        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projector_momentum = copy.deepcopy(self.projector)
        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projector_momentum)

    def forward_view(self, view_sample):
        """Foward one view sample to get the output."""
        encoded_features = self.encoder(view_sample).flatten(start_dim=1)
        projected_features = self.projector(encoded_features)
        prediction = self.predictor(projected_features)
        return projected_features, prediction

    def forward_momentum(self, view_sample):
        """Foward one view sample to get the output in a momentum manner."""
        features = self.encoder_momentum(view_sample).flatten(start_dim=1)
        encoded = self.projector_momentum(features)
        return encoded

    def forward(self, multiview_samples):
        """Forward the multiview samples."""
        view1, view2 = multiview_samples

        if self.n_iteration % 2:
            # Swap batches every two iterations.
            view2, view1 = view1, view2

        view1_encoded, view1_predicted = self.forward_view(view1)

        view2_encoded = self.forward_momentum(view2)

        # Update group features and get group assignments.
        assignments = self.smog.assign_groups(view2_encoded)
        group_features = self.smog.get_updated_group_features(view1_encoded)
        logits = self.smog(
            view1_predicted, group_features, temperature=self.temperature
        )
        self.smog.set_group_features(group_features)

        return logits, assignments
