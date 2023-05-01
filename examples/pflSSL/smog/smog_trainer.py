"""
The implemetation of the trainer for SMoG approach.
"""

from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.utils import update_momentum

from plato.trainers import basic_ssl
from plato.trainers import loss_criterion
from lightly.utils.scheduler import cosine_schedule

from plato.config import Config


class Trainer(basic_ssl.Trainer):
    """A personalized federated learning trainer with self-supervised learning."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        self.momentum_val = 0

        # training steps
        self.global_step = 0
        self.local_step = 0

        # memory bank because we reset the group features every 300 iterations
        self.reset_interval = (
            Config().trainer.reset_interval
            if hasattr(Config().trainer, "reset_interval")
            else 300
        )
        memory_bank_size = self.reset_interval * Config().trainer.batch_size
        self.memory_bank = MemoryBankModule(size=memory_bank_size)

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform training."""
        self.optimizer.zero_grad()

        outputs = self.model(examples)

        encoded_samples = outputs[-1]

        loss = self._loss_criterion(outputs[:2], labels)
        self._loss_tracker.update(loss, labels.size(0))

        self.memory_bank(encoded_samples, update=True)

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss

    def train_epoch_start(self, config):
        """Operations before starting one epoch."""
        super().train_epoch_start(config)
        epoch = self.current_epoch
        total_epochs = config["epochs"] * config["rounds"]
        self.momentum_val = cosine_schedule(epoch, total_epochs, 0.996, 1)

    def train_step_start(self, config, batch=None):
        """Operations before starting one iteration."""
        super().train_step_start(config)

        # update the global step
        self.global_step += batch

        if self.global_step > 0 and self.global_step % self.reset_interval == 0:
            # reset group features and weights every 300 iterations
            self.model.reset_group_features(memory_bank=self.memory_bank)
            self.model.reset_momentum_weights()
        else:
            update_momentum(
                self.model.encoder, self.model.encoder_momentum, m=self.momentum_val
            )
            update_momentum(
                self.model.projection_head,
                self.model.projection_head_momentum,
                m=self.momentum_val,
            )

        # update the local iteration
        self.model.n_iteration = batch
