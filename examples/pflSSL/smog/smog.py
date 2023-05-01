"""
The implementation for the BYOL [1] method.

[1]. Jean-Bastien Grill, et.al, Bootstrap Your Own Latent A New Approach to Self-Supervised Learning.
https://arxiv.org/pdf/2006.07733.pdf.

Source code: https://github.com/lucidrains/byol-pytorch
The third-party code: https://github.com/sthalles/PyTorch-BYOL
"""
import copy

import torch
from torch import nn
from sklearn.cluster import KMeans
from lightly.models.modules.heads import (
    SMoGPredictionHead,
    SMoGProjectionHead,
    SMoGPrototypes,
)
from lightly.models.utils import deactivate_requires_grad


from plato.servers import fedavg_personalized
from plato.clients import simple_ssl
from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config

from smog_trainer import Trainer


class SMoGModel(nn.Module):
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

        self.n_groups = 300
        self.smog = SMoGPrototypes(
            group_features=torch.rand(self.n_groups, 128), beta=0.99
        )

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
        samples1_encoded, samples1_predicted = self.forward_direct(samples1)

        samples2_encoded = self.forward_momentum(samples2)

        # update group features and get group assignments
        assignments = self.smog.assign_groups(samples2_encoded)
        group_features = self.smog.get_updated_group_features(samples1_encoded)
        logits = self.smog(samples1_predicted, group_features, temperature=self.temperature)
        self.smog.set_group_features(group_features)

        return logits, assignments


def main():
    """A Plato federated learning training session using the BYOL algorithm."""

    trainer = Trainer
    client = simple_ssl.Client(model=SMoGModel, trainer=trainer)
    server = fedavg_personalized.Server(model=SMoGModel, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
