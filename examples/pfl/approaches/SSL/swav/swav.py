"""
The implementation for the SwAV [1] method.

[1]. Mathilde Caron, et.al, Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.
https://arxiv.org/abs/2006.09882, NeurIPS 2020.

Source code: https://github.com/facebookresearch/swav
"""

from torch import nn
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes

from plato.trainers import loss_criterion
from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config

from pflbases import fedavg_personalized_server
from pflbases import fedavg_partial

from pflbases.trainer_callbacks import separate_trainer_callbacks
from pflbases.trainer_callbacks import ssl_trainer_callbacks
from pflbases.client_callbacks import local_completion_callbacks

from pflbases import ssl_client
from pflbases import ssl_trainer
from pflbases import ssl_datasources


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


def main():
    """
    A pFL sesstion for SwaV approach.
    """
    trainer = ssl_trainer.Trainer
    client = ssl_client.Client(
        model=SwaV,
        datasource=ssl_datasources.TransformedDataSource,
        personalized_datasource=ssl_datasources.TransformedDataSource,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
        callbacks=[
            local_completion_callbacks.ClientModelLocalCompletionCallback,
        ],
        trainer_callbacks=[
            separate_trainer_callbacks.PersonalizedModelMetricCallback,
            separate_trainer_callbacks.PersonalizedModelStatusCallback,
            ssl_trainer_callbacks.ModelStatusCallback,
        ],
    )
    server = fedavg_personalized_server.Server(
        model=SwaV,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
