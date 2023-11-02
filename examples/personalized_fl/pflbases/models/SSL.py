"""
Different SSL methods.
"""
import copy

import torch
from torch import nn

from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from sklearn.cluster import KMeans
from lightly.models.modules.heads import (
    SMoGPredictionHead,
    SMoGProjectionHead,
    SMoGPrototypes,
)
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.modules import MoCoProjectionHead

from lightly.models.utils import deactivate_requires_grad

from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config


class SimSiam(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()

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

        self.projection_head = SimSiamProjectionHead(
            self.encoder.encoding_dim,
            Config().trainer.projection_hidden_dimm,
            Config().trainer.projection_out_dim,
        )
        self.prediction_head = SimSiamPredictionHead(
            Config().trainer.projection_out_dim,
            Config().trainer.prediction_hidden_dim,
            Config().trainer.prediction_out_dim,
        )

    def forward_direct(self, samples):
        encoded_samples = self.encoder(samples).flatten(start_dim=1)
        projected_samples = self.projection_head(encoded_samples)
        output = self.prediction_head(projected_samples)
        projected_samples = projected_samples.detach()
        return projected_samples, output

    def forward(self, multiview_samples):
        samples1, samples2 = multiview_samples
        projected_samples1, output1 = self.forward_direct(samples1)
        projected_samples2, output2 = self.forward_direct(samples2)
        return (projected_samples1, output2), (projected_samples2, output1)


class SMoG(nn.Module):
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
        self.smog = SMoGPrototypes(
            group_features=torch.rand(self.n_groups, n_prototypes), beta=beta
        )

        # Current iteration.
        self.n_iteration = 0

    def _cluster_features(self, features: torch.Tensor) -> torch.Tensor:
        # Cluster the features using sklearn
        # (note: faiss is probably more efficient).
        features = features.cpu().numpy()
        kmeans = KMeans(self.n_groups).fit(features)
        clustered = torch.from_numpy(kmeans.cluster_centers_).float()
        clustered = torch.nn.functional.normalize(clustered, dim=1)
        return clustered

    def reset_group_features(self, memory_bank):
        # See https://arxiv.org/pdf/2207.06167.pdf Table 7b).
        features = memory_bank.bank
        group_features = self._cluster_features(features.t())
        self.smog.set_group_features(group_features)

    def reset_momentum_weights(self):
        # See https://arxiv.org/pdf/2207.06167.pdf Table 7b).
        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward_direct(self, samples):
        features = self.encoder(samples).flatten(start_dim=1)
        encoded = self.projection_head(features)
        predicted = self.prediction_head(encoded)
        return encoded, predicted

    def forward_momentum(self, samples):
        features = self.encoder_momentum(samples).flatten(start_dim=1)
        encoded = self.projection_head_momentum(features)
        return encoded

    def forward(self, multiview_samples):
        samples1, samples2 = multiview_samples

        if self.n_iteration % 2:
            # Swap batches every two iterations.
            samples2, samples1 = samples1, samples2

        samples1_encoded, samples1_predicted = self.forward_direct(samples1)

        samples2_encoded = self.forward_momentum(samples2)

        # Update group features and get group assignments.
        assignments = self.smog.assign_groups(samples2_encoded)
        group_features = self.smog.get_updated_group_features(samples1_encoded)
        logits = self.smog(
            samples1_predicted, group_features, temperature=self.temperature
        )
        self.smog.set_group_features(group_features)

        return logits, assignments, samples1_encoded


class SwaV(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()

        # Define the encoder.
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )

        # Define the encoder.
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = encoder_registry.get(
                model_name=encoder_name, **encoder_params
            )

        self.encoding_dim = self.encoder.encoding_dim

        # Define the projector.
        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim
        n_prototypes = Config().trainer.n_prototypes

        self.projection_head = SwaVProjectionHead(
            self.encoding_dim, projection_hidden_dim, projection_out_dim
        )
        self.prototypes = SwaVPrototypes(projection_out_dim, n_prototypes=n_prototypes)

    def forward_direct(self, samples):
        encoded_samples = self.encoder(samples).flatten(start_dim=1)
        encoded_samples = self.projection_head(encoded_samples)
        encoded_samples = nn.functional.normalize(encoded_samples, dim=1, p=2)
        outputs = self.prototypes(encoded_samples)
        return outputs

    def forward(self, multiview_samples):
        self.prototypes.normalize()
        multi_crop_features = [
            self.forward_direct(sample) for sample in multiview_samples
        ]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]

        return high_resolution, low_resolution


class MoCoV2(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()

        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )
        # Define the encoder.
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = encoder_registry.get(
                model_name=encoder_name, **encoder_params
            )

        self.encoding_dim = self.encoder.encoding_dim

        # Define heads.
        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim
        self.projection_head = MoCoProjectionHead(
            self.encoding_dim, projection_hidden_dim, projection_out_dim
        )

        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward_direct(self, samples):
        query = self.encoder(samples).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, samples):
        key = self.encoder_momentum(samples).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def forward(self, multiview_samples):
        query_samples, key_samples = multiview_samples
        query = self.forward_direct(query_samples)
        key = self.forward_momentum(key_samples)

        return query, key
