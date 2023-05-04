"""
The implementation for the MoCoV2 [2] method, which is the enhanced version of MoCoV1 [1],
for personalized federated learning.

[1]. Kaiming He, et.al., Momentum Contrast for Unsupervised Visual Representation Learning, 
CVPR 2020. https://arxiv.org/abs/1911.05722.

[2]. Xinlei Chen, et.al, Improved Baselines with Momentum Contrastive Learning, ArXiv, 2020.
https://arxiv.org/abs/2003.04297.

The official code: https://github.com/facebookresearch/moco


"""

import copy

from torch import nn
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from examples.pfl.bases import fedavg_personalized
from plato.trainers import basic_ssl
from examples.pfl.bases import simple_ssl
from plato.models.cnn_encoder import Model as encoder_registry
from plato.config import Config


class Trainer(basic_ssl.Trainer):
    """A personalized federated learning trainer with self-supervised learning."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        self.momentum_val = 0

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


def main():
    """A Plato personalized federated learning training session using the SimCLR approach."""

    trainer = Trainer
    client = simple_ssl.Client(model=MoCoV2, trainer=trainer)
    server = fedavg_personalized.Server(model=MoCoV2, trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
