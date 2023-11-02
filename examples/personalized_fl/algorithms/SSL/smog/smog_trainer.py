"""
The implemetation of the trainer for SMoG approach.
"""
import os

import torch
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from plato.config import Config

from pflbases import ssl_trainer


class Trainer(ssl_trainer.Trainer):
    """A base trainer for SMoG approach."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        self.momentum_val = 0

        # Set training steps
        self.global_step = 0
        self.local_step = 0

        # Set the memory bank because we
        # reset the group features every 300 iterations
        self.reset_interval = (
            Config().trainer.reset_interval
            if hasattr(Config().trainer, "reset_interval")
            else 300
        )
        self.memory_bank = MemoryBankModule(
            size=self.reset_interval * Config().trainer.batch_size
        )

    def model_forward(self, examples):
        """Forward the input examples to the model."""
        outputs = self.model(examples)
        encoded_samples = outputs[-1]
        self.memory_bank(encoded_samples, update=True)
        return outputs[:2]

    def train_epoch_start(self, config):
        """Operation before starting one epoch."""
        super().train_epoch_start(config)
        epoch = self.current_epoch
        total_epochs = config["epochs"] * config["rounds"]
        # Update the momentum value for the current epoch
        # in regular federated training
        if not self.current_round > Config().trainer.rounds:
            self.momentum_val = cosine_schedule(epoch, total_epochs, 0.996, 1)

    def train_step_start(self, config, batch=None):
        """Operation before starting one iteration."""
        super().train_step_start(config)

        if not self.current_round > Config().trainer.rounds:
            # Update the global step
            self.global_step += batch

            if self.global_step > 0 and self.global_step % self.reset_interval == 0:
                # Reset group features and weights every some iterations
                self.model.reset_group_features(memory_bank=self.memory_bank)
                self.model.reset_momentum_weights()
            else:
                update_momentum(
                    self.model.encoder, self.model.encoder_momentum, m=self.momentum_val
                )
                update_momentum(
                    self.model.projector,
                    self.model.projector_momentum,
                    m=self.momentum_val,
                )

            # update the local iteration
            self.model.n_iteration = batch

    def train_run_start(self, config):
        """Load the memory bank from file system."""
        super().train_run_start(config)
        if not self.current_round > Config().trainer.rounds:
            model_path = Config().params["model_path"]
            filename_bank = f"client_{self.client_id}_bank.pth"
            filename_ptr = f"client_{self.client_id}_ptr.pth"
            bank_path = os.path.join(model_path, filename_bank)
            ptr_path = os.path.join(model_path, filename_ptr)

            if os.path.exists(bank_path):
                self.memory_bank.bank = torch.load(bank_path)
                self.memory_bank.bank_ptr = torch.load(ptr_path)

    def train_run_end(self, config):
        """Save the memory bank to the file system."""
        super().train_run_end(config)
        if not self.current_round > Config().trainer.rounds:
            model_path = Config().params["model_path"]
            filename_bank = f"client_{self.client_id}_bank.pth"
            filename_ptr = f"client_{self.client_id}_ptr.pth"

            bank_path = os.path.join(model_path, filename_bank)
            ptr_path = os.path.join(model_path, filename_ptr)
            torch.save(self.memory_bank.bank, bank_path)
            torch.save(self.memory_bank.bank_ptr, ptr_path)
