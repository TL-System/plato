"""
The implementation for the BYOL [1] method.

[1]. Jean-Bastien Grill, et.al, Bootstrap Your Own Latent A New Approach to Self-Supervised Learning.
https://arxiv.org/pdf/2006.07733.pdf.

Source code: https://github.com/lucidrains/byol-pytorch
The third-party code: https://github.com/sthalles/PyTorch-BYOL
"""
import copy

from torch import nn
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

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


class Trainer(ssl_trainer.Trainer):
    """A trainer for BYOL to rewrite the loss wrappe."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        self.momentum_val = 0

    def plato_ssl_loss_wrapper(self):
        """A wrapper to connect ssl loss with plato."""
        defined_ssl_loss = loss_criterion.get()
        epochs = Config.trainer.epochs
        gloabl_epoch = (self.current_round - 1) * epochs + self.current_epoch

        def compute_plato_loss(outputs, labels):
            if isinstance(outputs, (list, tuple)):
                loss = defined_ssl_loss(*outputs, gloabl_epoch)
                return loss
            else:
                return defined_ssl_loss(outputs)

        return compute_plato_loss

    def loss_backward_end(self):
        """Stopping gradients for final layer of the model."""
        if not self.personalized_learning:
            self.model.projection_head.cancel_last_layer_gradients(
                current_epoch=self.current_epoch
            )

    def train_epoch_start(self, config):
        """Operations before starting one epoch."""
        super().train_epoch_start(config)
        epoch = self.current_epoch
        total_epochs = config["epochs"] * config["rounds"]
        global_epoch = (self.current_round - 1) * config["epochs"] + epoch
        if not self.personalized_learning:
            self.momentum_val = cosine_schedule(global_epoch, total_epochs, 0.996, 1)

    def train_step_start(self, config, batch=None):
        """Operations before starting one iteration."""
        super().train_step_start(config)
        if not self.personalized_learning:
            update_momentum(
                self.model.encoder, self.model.teacher_encoder, m=self.momentum_val
            )
            update_momentum(
                self.model.projection_head,
                self.model.teacher_head,
                m=self.momentum_val,
            )


class DINO(nn.Module):
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
        projection_bottleneck_dim = Config().trainer.projection_bottleneck_dim
        projection_out_dim = Config().trainer.projection_out_dim

        self.projection_head = DINOProjectionHead(
            self.encoding_dim,
            projection_hidden_dim,
            projection_bottleneck_dim,
            projection_out_dim,
        )
        # Detach the weights from the computation graph
        self.projection_head.last_layer.weight_g.detach_()
        self.projection_head.last_layer.weight_g.requires_grad = False

        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_head = DINOProjectionHead(
            self.encoding_dim,
            projection_hidden_dim,
            projection_bottleneck_dim,
            projection_out_dim,
        )
        self.teacher_head.last_layer.weight_g.detach_()
        self.teacher_head.last_layer.weight_g.requires_grad = False

        deactivate_requires_grad(self.teacher_encoder)
        deactivate_requires_grad(self.teacher_head)

    def forward_student(self, samples):
        encoded_examples = self.encoder(samples).flatten(start_dim=1)
        projected_examples = self.projection_head(encoded_examples)
        return projected_examples

    def forward_teacher(self, samples):
        encoded_examples = self.teacher_encoder(samples).flatten(start_dim=1)
        projected_examples = self.teacher_head(encoded_examples)
        return projected_examples

    def forward(self, multiview_samples):
        global_views = multiview_samples[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward_student(view) for view in global_views]
        return teacher_out, student_out


def main():
    """
    A personalized federated learning sesstion for BYOL approach.
    """
    trainer = Trainer
    client = ssl_client.Client(
        model=DINO,
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
        model=DINO,
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )

    server.run(client)


if __name__ == "__main__":
    main()
