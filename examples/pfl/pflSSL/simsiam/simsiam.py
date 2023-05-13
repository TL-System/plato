"""
The implementation for the SimSiam [1] method.

[1]. Xinlei Chen, et.al, Exploring Simple Siamese Representation Learning.
https://arxiv.org/pdf/2011.10566.pdf

Source code: https://github.com/facebookresearch/simsiam
Third-party code: https://github.com/PatrickHua/SimSiam
"""


import os
import sys

# Add `bases` to the path
pfl_bases = os.path.dirname(os.path.abspath(__file__))
grandparent_directory = os.path.abspath(os.path.join(pfl_bases, os.pardir, os.pardir))
sys.path.insert(1, grandparent_directory)

from torch import nn

from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead

from plato.trainers import loss_criterion

from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config

from bases import fedavg_personalized_server
from bases import fedavg_partial

from bases.trainer_callbacks import separate_trainer_callbacks
from bases.trainer_callbacks import ssl_trainer_callbacks
from bases.client_callbacks import local_completion_callbacks

from bases import ssl_client
from bases import ssl_trainer
from bases import ssl_datasources


class Trainer(ssl_trainer.Trainer):
    """A personalized federated learning trainer with self-supervised learning."""

    def plato_ssl_loss_wrapper(self):
        """A wrapper to connect ssl loss with plato."""
        defined_ssl_loss = loss_criterion.get()

        def compute_plato_loss(outputs, labels):
            if isinstance(outputs, (list, tuple)):
                loss = 0.5 * (
                    defined_ssl_loss(*outputs[0]) + defined_ssl_loss(*outputs[1])
                )
                return loss
            else:
                return defined_ssl_loss(outputs)

        return compute_plato_loss


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


def main():
    """
    A Plato personalized federated learning sesstion for FedBABU approach.
    """
    trainer = Trainer
    client = ssl_client.Client(
        model=SimSiam,
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
        model=SimSiam,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
