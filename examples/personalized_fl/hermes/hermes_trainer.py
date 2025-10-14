"""
The trainer used by clients using Hermes.

This trainer uses the composable trainer architecture with a custom callback
to implement structured pruning for personalized federated learning.
"""

import logging
import os
import pickle

import hermes_pruning as pruning
import torch
from torch.nn.utils import prune

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.trainers import basic


class HermesPruningCallback(TrainerCallback):
    """
    Callback implementing Hermes structured pruning functionality.

    Handles:
    - Evaluating model accuracy to decide whether to prune
    - Conducting structured pruning based on accuracy threshold
    - Applying and saving pruning masks
    - Making pruning permanent after training
    """

    def __init__(self):
        """Initialize Hermes pruning state."""
        self.mask = None
        self.pruning_target = Config().clients.pruning_target * 100
        self.pruned_amount = 0
        self.pruning_rate = Config().clients.pruning_amount * 100
        self.datasource = None
        self.testset = None
        self.need_prune = False
        self.accuracy_threshold = (
            Config().clients.accuracy_threshold
            if hasattr(Config().clients, "accuracy_threshold")
            else 0.5
        )

    def on_train_run_start(self, trainer, config, **kwargs):
        """Conduct pruning if needed before training."""
        # Evaluate if structured pruning should be conducted
        self.datasource = datasources_registry.get(client_id=trainer.client_id)
        self.testset = self.datasource.get_test_set()
        logging.info(
            "[Client #%d] Testing the model for prune decision.", trainer.client_id
        )
        accuracy = trainer.test_model(config, self.testset, None)
        self.pruned_amount = pruning.compute_pruned_amount(
            trainer.model, trainer.client_id
        )

        # Apply the mask to the incoming server payload model to create the model for training
        trainer.model = self._apply_mask(trainer, trainer.model)

        # Send the model to the device used for training
        trainer.model.to(trainer.device)
        trainer.model.train()

        logging.info(
            "[Client #%d] Evaluated Accuracy for pruning: %.2f%%",
            trainer.client_id,
            accuracy * 100,
        )

        if (
            accuracy >= self.accuracy_threshold
            and self.pruned_amount < self.pruning_target
        ):
            logging.info(
                "[Client #%d] Conducting structured pruning.", trainer.client_id
            )

            if self.pruning_target - self.pruned_amount < self.pruning_rate:
                self.pruning_rate = (self.pruning_target - self.pruned_amount) / 100
                self.mask = pruning.structured_pruning(
                    trainer.model, self.pruning_rate, adjust_rate=self.pruned_amount
                )
            else:
                self.pruning_rate = (self.pruning_rate) / (100 - self.pruned_amount)
                self.mask = pruning.structured_pruning(
                    trainer.model,
                    self.pruning_rate,
                )

            self._save_mask(trainer, self.mask)
            self.need_prune = True
        else:
            logging.info("[Client #%d] No need to prune.", trainer.client_id)
            self.need_prune = False

    def on_train_run_end(self, trainer, config, **kwargs):
        """Make pruning permanent if it was conducted."""
        if self.need_prune or self.pruned_amount > 0:
            for __, layer in trainer.model.named_parameters():
                if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                    prune.remove(layer, "weight")

            self.pruned_amount = pruning.compute_pruned_amount(
                trainer.model, trainer.client_id
            )
            logging.info(
                "[Client #%d] Pruned Amount: %.2f%%",
                trainer.client_id,
                self.pruned_amount,
            )

    def _apply_mask(self, trainer, model):
        """Apply the mask onto the incoming personalized model."""
        model_name = Config().trainer.model_name
        model_path = Config().params["model_path"]

        mask_path = f"{model_path}/{model_name}_client{trainer.client_id}_mask.pth"
        if not os.path.exists(mask_path):
            return model
        else:
            with open(mask_path, "rb") as mask_file:
                mask = pickle.load(mask_file)

        return pruning.apply_mask(model, mask, trainer.device)

    def _save_mask(self, trainer, mask):
        """Save the mask for merging in future rounds if pruning has occurred."""
        model_name = Config().trainer.model_name
        model_path = Config().params["model_path"]

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        mask_path = f"{model_path}/{model_name}_client{trainer.client_id}_mask.pth"

        with open(mask_path, "wb") as payload_file:
            pickle.dump(mask, payload_file)


class Trainer(basic.Trainer):
    """
    A federated learning trainer using Hermes algorithm.

    This trainer implements structured pruning for personalized federated learning.
    It evaluates model accuracy at the start of each training round and conducts
    pruning if the accuracy threshold is met and the pruning target is not yet reached.

    It uses the composable trainer architecture with a callback for pruning management.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the Hermes trainer with pruning callback.

        Arguments:
            model: The model to train
            callbacks: List of callback classes or instances
        """
        # Create Hermes pruning callback
        hermes_callback = HermesPruningCallback()

        # Combine with provided callbacks
        all_callbacks = [hermes_callback]
        if callbacks is not None:
            all_callbacks.extend(callbacks)

        # Initialize parent trainer with combined callbacks
        super().__init__(model=model, callbacks=all_callbacks)
