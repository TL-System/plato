"""
The implementation for the SimCLR [1] method for personalized federated learning.

[1]. Ting Chen, et.al., A Simple Framework for Contrastive Learning of Visual Representations, 
ICML 2020. https://arxiv.org/abs/2002.05709

The official code: https://github.com/google-research/simclr

The structure of our SimCLR and the classifier is the same as the ones used in
the work https://github.com/spijkervet/SimCLR.git.

"""


import os
import sys

# Add `bases` to the path
pfl_bases = os.path.dirname(os.path.abspath(__file__))
grandparent_directory = os.path.abspath(os.path.join(pfl_bases, os.pardir, os.pardir))
sys.path.insert(1, grandparent_directory)

import torch

from lightly.models.modules.heads import SimCLRProjectionHead

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


class SimCLR(torch.nn.Module):
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
    def get_model():
        return SimCLR()


def main():
    """
    A Plato personalized federated learning sesstion for FedBABU approach.
    """
    trainer = ssl_trainer.Trainer
    client = ssl_client.Client(
        model=SimCLR,
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
        model=SimCLR,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
