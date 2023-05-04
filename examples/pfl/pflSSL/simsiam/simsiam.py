"""
The implementation for the SimSiam [1] method.

[1]. Xinlei Chen, et.al, Exploring Simple Siamese Representation Learning.
https://arxiv.org/pdf/2011.10566.pdf

Source code: https://github.com/facebookresearch/simsiam
Third-party code: https://github.com/PatrickHua/SimSiam
"""

from torch import nn

from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead

from examples.pfl.bases import fedavg_personalized
from plato.trainers import basic_ssl
from examples.pfl.bases import simple_ssl
from plato.trainers import loss_criterion

from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config


class Trainer(basic_ssl.Trainer):
    """A personalized federated learning trainer with self-supervised learning."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        self.momentum_val = 0

    def get_loss_criterion(self):
        """Returns the loss criterion.
        As the loss functions derive from the lightly,
        it is desired to create a interface
        """

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
    """A Plato federated learning training session using the SimCLR algorithm.
    This implementation of simclr utilizes the general setting, i.e.,
    removing the final fully-connected layers of model defined by
    the 'model_name' in config file.
    """

    trainer = Trainer
    client = simple_ssl.Client(model=SimSiam, trainer=trainer)
    server = fedavg_personalized.Server(model=SimSiam, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
