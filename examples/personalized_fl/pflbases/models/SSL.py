"""
Different SSL methods.

"""
import copy

import torch
from torch import nn
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.modules.heads import SimCLRProjectionHead
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


class BYOL(nn.Module):
    def __init__(self, encoder=None, encoder_dim=None):
        super().__init__()

        # define the encoder
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )

        # define the encoder
        self.encoder = (
            encoder
            if encoder is not None
            else encoder_registry.get(model_name=encoder_name, **encoder_params)
        )

        self.encoding_dim = self.encoder.encoding_dim
        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim
        prediction_hidden_dim = Config().trainer.prediction_hidden_dim
        prediction_out_dim = Config().trainer.prediction_out_dim

        self.projection_head = BYOLProjectionHead(
            self.encoding_dim, projection_hidden_dim, projection_out_dim
        )
        self.prediction_head = BYOLPredictionHead(
            projection_out_dim, prediction_hidden_dim, prediction_out_dim
        )

        self.encoder_momentum = copy.deepcopy(self.encoder)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.encoder_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward_direct(self, samples):
        """Foward the samples to get the output."""
        encoded_examples = self.encoder(samples).flatten(start_dim=1)
        projected_examples = self.projection_head(encoded_examples)
        output = self.prediction_head(projected_examples)
        return output

    def forward_momentum(self, samples):
        """Foward the samples to get the output in a momentum manner."""
        encoded_examples = self.encoder_momentum(samples).flatten(start_dim=1)
        projected_examples = self.projection_head_momentum(encoded_examples)
        projected_examples = projected_examples.detach()
        return projected_examples

    def forward(self, multiview_samples):
        """Main forward function of the model."""
        samples1, samples2 = multiview_samples
        output1 = self.forward_direct(samples1)
        projected_samples1 = self.forward_momentum(samples1)
        output2 = self.forward_direct(samples2)
        projected_samples2 = self.forward_momentum(samples2)
        return (output1, projected_samples2), (output2, projected_samples1)

    @staticmethod
    def get():
        """Get the defined BYOL model."""
        return BYOL()


class SimCLR(nn.Module):
    """The model structure of SimCLR."""

    def __init__(self, encoder=None, encoder_dim=None):
        super().__init__()

        # extract hyper-parameters
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )
        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim

        # define the encoder based on the model_name in config
        self.encoder = (
            encoder
            if encoder is not None
            else encoder_registry.get(model_name=encoder_name, **encoder_params)
        )

        self.encoding_dim = self.encoder.encoding_dim
        self.projector = SimCLRProjectionHead(
            self.encoding_dim, projection_hidden_dim, projection_out_dim
        )

    def forward(self, multiview_samples):
        """Forward two batch of contrastive samples."""
        samples1, samples2 = multiview_samples
        encoded_h1 = self.encoder(samples1)
        encoded_h2 = self.encoder(samples2)

        projected_z1 = self.projector(encoded_h1)
        projected_z2 = self.projector(encoded_h2)
        return projected_z1, projected_z2

    @staticmethod
    def get():
        return SimCLR()


class SimSiam(nn.Module):
    def __init__(self, encoder=None, encoder_dim=None):
        super().__init__()

        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )
        # define the encoder based on the model_name in config
        self.encoder = (
            encoder
            if encoder is not None
            else encoder_registry.get(model_name=encoder_name, **encoder_params)
        )

        self.encoding_dim = self.encoder.encoding_dim

        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim
        prediction_hidden_dim = Config().trainer.prediction_hidden_dim
        prediction_out_dim = Config().trainer.prediction_out_dim

        self.projection_head = SimSiamProjectionHead(
            self.encoding_dim, projection_hidden_dim, projection_out_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            projection_out_dim, prediction_hidden_dim, prediction_out_dim
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

    @staticmethod
    def get():
        return SimSiam()


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
        # define the encoder
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )

        # define the encoder based on the model_name in config
        self.encoder = (
            encoder
            if encoder is not None
            else encoder_registry.get(model_name=encoder_name, **encoder_params)
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

        # current iteration
        self.n_iteration = 0

    def _cluster_features(self, features: torch.Tensor) -> torch.Tensor:
        # clusters the features using sklearn
        # (note: faiss is probably more efficient)
        features = features.cpu().numpy()
        kmeans = KMeans(self.n_groups).fit(features)
        clustered = torch.from_numpy(kmeans.cluster_centers_).float()
        clustered = torch.nn.functional.normalize(clustered, dim=1)
        return clustered

    def reset_group_features(self, memory_bank):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
        features = memory_bank.bank
        group_features = self._cluster_features(features.t())
        self.smog.set_group_features(group_features)

    def reset_momentum_weights(self):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
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
            # swap batches every two iterations
            samples2, samples1 = samples1, samples2

        samples1_encoded, samples1_predicted = self.forward_direct(samples1)

        samples2_encoded = self.forward_momentum(samples2)

        # update group features and get group assignments
        assignments = self.smog.assign_groups(samples2_encoded)
        group_features = self.smog.get_updated_group_features(samples1_encoded)
        logits = self.smog(
            samples1_predicted, group_features, temperature=self.temperature
        )
        self.smog.set_group_features(group_features)

        return logits, assignments, samples1_encoded

    @staticmethod
    def get():
        """Get the defined SMoG model."""
        return SMoG()


class SwaV(nn.Module):
    def __init__(self, encoder=None, encoder_dim=None):
        super().__init__()

        # define the encoder
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )

        # define the encoder
        self.encoder = (
            encoder
            if encoder is not None
            else encoder_registry.get(model_name=encoder_name, **encoder_params)
        )

        self.encoding_dim = self.encoder.encoding_dim

        # define the projector
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

    @staticmethod
    def get():
        """Get the defined SwaV model."""
        return SwaV()


class MoCoV2(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()

        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )
        # define the encoder
        self.encoder = (
            encoder
            if encoder is not None
            else encoder_registry.get(model_name=encoder_name, **encoder_params)
        )

        self.encoding_dim = self.encoder.encoding_dim

        # define heads
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

    @staticmethod
    def get():
        """Get the defined MoCoV2 model."""
        return MoCoV2()
