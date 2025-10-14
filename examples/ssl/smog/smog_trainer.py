"""
A self-supervised federated learning trainer with SMoG.

This trainer uses the composable trainer architecture with custom callbacks
to implement SMoG-specific functionality (momentum updates and memory bank management).
"""

import os

import torch
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers import self_supervised_learning as ssl_trainer


class SMoGCallback(TrainerCallback):
    """
    Callback implementing SMoG algorithm functionality.

    Handles:
    - Momentum value computation and model updates
    - Memory bank management (loading, saving, resetting)
    - Global step tracking
    """

    def __init__(self):
        """Initialize SMoG-specific state."""
        # The momentum value used to update the model with Exponential Moving Average
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

    def on_train_run_start(self, trainer, config, **kwargs):
        """Load the memory bank from file system at the start of training."""
        # Load the memory bank from the file system during regular federated training
        if not trainer.current_round > Config().trainer.rounds:
            model_path = Config().params["model_path"]
            filename_bank = f"client_{trainer.client_id}_bank.pth"
            filename_ptr = f"client_{trainer.client_id}_ptr.pth"
            bank_path = os.path.join(model_path, filename_bank)
            ptr_path = os.path.join(model_path, filename_ptr)

            if os.path.exists(bank_path):
                self.memory_bank.bank = torch.load(bank_path)
                self.memory_bank.bank_ptr = torch.load(ptr_path)

    def on_train_run_end(self, trainer, config, **kwargs):
        """Save the memory bank to the file system at the end of training."""
        # Save the memory bank to the file system during regular federated training
        if not trainer.current_round > Config().trainer.rounds:
            model_path = Config().params["model_path"]
            filename_bank = f"client_{trainer.client_id}_bank.pth"
            filename_ptr = f"client_{trainer.client_id}_ptr.pth"

            bank_path = os.path.join(model_path, filename_bank)
            ptr_path = os.path.join(model_path, filename_ptr)
            torch.save(self.memory_bank.bank, bank_path)
            torch.save(self.memory_bank.bank_ptr, ptr_path)

    def on_train_epoch_start(self, trainer, config, **kwargs):
        """Compute the momentum value before starting one epoch of training."""
        epoch = trainer.current_epoch
        total_epochs = config["epochs"] * config["rounds"]

        # Update the momentum value for the current epoch in regular federated training
        if not trainer.current_round > Config().trainer.rounds:
            self.momentum_val = cosine_schedule(epoch, total_epochs, 0.996, 1)

    def on_train_step_start(self, trainer, config, batch, **kwargs):
        """
        Update the model based on the computed momentum value in each training step.
        Reset the memory bank when it reaches full size.
        """
        if not trainer.current_round > Config().trainer.rounds:
            # Update the global step
            self.global_step += 1

            if self.global_step > 0 and self.global_step % self.reset_interval == 0:
                # Reset group features and momentum weights when the memory bank is full
                trainer.model.reset_group_features(memory_bank=self.memory_bank)
                trainer.model.reset_momentum_weights()
            else:
                # Update the model based on the momentum value
                # Specifically, it updates parameters of `encoder` with
                # Exponential Moving Average of `encoder_momentum`
                update_momentum(
                    trainer.model.encoder,
                    trainer.model.encoder_momentum,
                    m=self.momentum_val,
                )
                update_momentum(
                    trainer.model.projector,
                    trainer.model.projector_momentum,
                    m=self.momentum_val,
                )

            # Update the local iteration for the model
            trainer.model.n_iteration = batch


class Trainer(ssl_trainer.Trainer):
    """
    A federated learning trainer using SMoG algorithm.

    This trainer extends the SSL trainer with SMoG-specific functionality
    via a custom callback. It uses the composable trainer architecture with
    callbacks for momentum updates and memory bank management.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the SMoG trainer with SMoG callback.

        Arguments:
            model: The model to train (SSL model with momentum encoders)
            callbacks: List of callback classes or instances
        """
        # Create SMoG callback
        smog_callback = SMoGCallback()

        # Combine with provided callbacks
        all_callbacks = [smog_callback]
        if callbacks is not None:
            all_callbacks.extend(callbacks)

        # Initialize parent SSL trainer with combined callbacks
        super().__init__(model=model, callbacks=all_callbacks)
