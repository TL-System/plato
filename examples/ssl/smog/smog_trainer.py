"""
A self-supervised federated learning trainer with SMoG.
"""
import os

import torch
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from plato.trainers import self_supervised_learning as ssl_trainer
from plato.config import Config


class Trainer(ssl_trainer.Trainer):
    """
    A trainer with SMoG, which computes the momentum value to update the model
    in each training step and loads the 'memory_bank' to facilitate the model forward.
    After training, the 'memory_bank' from the trained model will be saved to disk for
    subsequent learning.
    """

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # The momentum value used to update the model
        # with Exponential Moving Average
        self.momentum_val = 0

        # Set training steps
        self.global_step = 0

        # Set the memory bank and its size
        # The reset_interval used here is the common term to show
        # how many iterations we reset this memory bank.
        # The number used by the authors is 300
        self.reset_interval = (
            Config().trainer.reset_interval
            if hasattr(Config().trainer, "reset_interval")
            else 300
        )
        self.memory_bank = MemoryBankModule(
            size=self.reset_interval * Config().trainer.batch_size
        )

    def train_epoch_start(self, config):
        """Compute the momentum value before starting one epoch of training."""
        super().train_epoch_start(config)
        epoch = self.current_epoch
        total_epochs = config["epochs"] * config["rounds"]
        # Update the momentum value for the current epoch
        # in regular federated training
        if not self.current_round > Config().trainer.rounds:
            self.momentum_val = cosine_schedule(epoch, total_epochs, 0.996, 1)

    def train_step_start(self, config, batch=None):
        """
        Update the model based on the computed momentum value in each training step.
        And reset the 'memory bank' along with all momentum values when the number of
        collected features in this bank reaches its full size.
        """
        super().train_step_start(config)

        if not self.current_round > Config().trainer.rounds:
            # Update the global step
            self.global_step += 1

            if self.global_step > 0 and self.global_step % self.reset_interval == 0:
                # Reset group features and momentum weights when the memory bank is
                # full, i.e., the number of features added to the bank
                # in each iteration step, reaches its full size.
                self.model.reset_group_features(memory_bank=self.memory_bank)
                self.model.reset_momentum_weights()
            else:
                # Update the model based on the momentum value
                # Specifically, it updates parameters of `encoder` with
                # Exponential Moving Average of `encoder_momentum`
                update_momentum(
                    self.model.encoder, self.model.encoder_momentum, m=self.momentum_val
                )
                update_momentum(
                    self.model.projector,
                    self.model.projector_momentum,
                    m=self.momentum_val,
                )

            # Update the local iteration for the model
            self.model.n_iteration = batch

    def train_run_start(self, config):
        """Load the memory bank from file system."""
        super().train_run_start(config)
        # Load the memory bank from the file system during
        # regular federated training
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
        # Load the memory bank from the file system during
        # regular federated training
        if not self.current_round > Config().trainer.rounds:
            model_path = Config().params["model_path"]
            filename_bank = f"client_{self.client_id}_bank.pth"
            filename_ptr = f"client_{self.client_id}_ptr.pth"

            bank_path = os.path.join(model_path, filename_bank)
            ptr_path = os.path.join(model_path, filename_ptr)
            torch.save(self.memory_bank.bank, bank_path)
            torch.save(self.memory_bank.bank_ptr, ptr_path)
