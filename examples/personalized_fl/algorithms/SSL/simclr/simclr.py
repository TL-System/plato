"""
The implementation for the SimCLR [1] method for personalized federated learning.

[1]. Ting Chen, et al., A Simple Framework for Contrastive Learning of Visual Representations, 
ICML 2020. https://arxiv.org/abs/2002.05709

The official code: https://github.com/google-research/simclr

The structure of our SimCLR and the classifier is the same as the ones used in
the work https://github.com/spijkervet/SimCLR.git.

"""
from lightly.models.modules.heads import SimCLRProjectionHead
from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config
from torch import nn

from pflbases import fedavg_personalized

from pflbases import ssl_datasources
from pflbases import ssl_client
from pflbases import ssl_trainer


class SimCLR(nn.Module):
    """The model structure of SimCLR."""

    def __init__(self, encoder=None):
        super().__init__()

        # Extract hyper-parameters.
        encoder_name = Config().trainer.encoder_name
        encoder_params = (
            Config().params.encoder if hasattr(Config().params, "encoder") else {}
        )
        projection_hidden_dim = Config().trainer.projection_hidden_dim
        projection_out_dim = Config().trainer.projection_out_dim

        # Define the encoder based on the model_name in config.
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = encoder_registry.get(
                model_name=encoder_name, **encoder_params
            )

        self.projector = SimCLRProjectionHead(
            self.encoder.encoding_dim, projection_hidden_dim, projection_out_dim
        )

    def forward(self, multiview_samples):
        """Forward two batch of contrastive samples."""
        samples1, samples2 = multiview_samples
        encoded_h1 = self.encoder(samples1)
        encoded_h2 = self.encoder(samples2)

        projected_z1 = self.projector(encoded_h1)
        projected_z2 = self.projector(encoded_h2)
        return projected_z1, projected_z2


def main():
    """
    A personalized federated learning session for SimCLR approach.
    """
    trainer = ssl_trainer.Trainer
    client = ssl_client.Client(
        model=SimCLR,
        datasource=ssl_datasources.SSLDataSource,
        trainer=trainer,
    )
    server = fedavg_personalized.Server(model=SimCLR, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
