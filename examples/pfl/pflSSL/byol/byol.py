"""
The implementation for the BYOL [1] method.

[1]. Jean-Bastien Grill, et.al, Bootstrap Your Own Latent A New Approach to Self-Supervised Learning.
https://arxiv.org/pdf/2006.07733.pdf.

Source code: https://github.com/lucidrains/byol-pytorch
The third-party code: https://github.com/sthalles/PyTorch-BYOL
"""
import copy

from torch import nn
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

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

    def train_epoch_start(self, config):
        """Operations before starting one epoch."""
        super().train_epoch_start(config)
        epoch = self.current_epoch
        total_epochs = config["epochs"] * config["rounds"]
        self.momentum_val = cosine_schedule(epoch, total_epochs, 0.996, 1)

    def train_step_start(self, config, batch=None):
        """Operations before starting one iteration."""
        super().train_step_start(config)
        update_momentum(
            self.model.encoder, self.model.encoder_momentum, m=self.momentum_val
        )
        update_momentum(
            self.model.projection_head,
            self.model.projection_head_momentum,
            m=self.momentum_val,
        )


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
        encoded_examples = self.encoder(samples).flatten(start_dim=1)
        projected_examples = self.projection_head(encoded_examples)
        output = self.prediction_head(projected_examples)
        return output

    def forward_momentum(self, samples):
        encoded_examples = self.encoder_momentum(samples).flatten(start_dim=1)
        projected_examples = self.projection_head_momentum(encoded_examples)
        projected_examples = projected_examples.detach()
        return projected_examples

    def forward(self, multiview_samples):
        samples1, samples2 = multiview_samples
        output1 = self.forward_direct(samples1)
        projected_samples1 = self.forward_momentum(samples1)
        output2 = self.forward_direct(samples2)
        projected_samples2 = self.forward_momentum(samples2)
        return (output1, projected_samples2), (output2, projected_samples1)


def main():
    """A Plato federated learning training session using the BYOL algorithm."""

    trainer = Trainer
    client = simple_ssl.Client(model=BYOL, trainer=trainer)
    server = fedavg_personalized.Server(model=BYOL, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
