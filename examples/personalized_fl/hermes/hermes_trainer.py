"""
The trainer used by clients using Hermes.
"""

import logging
import os

import pickle

import torch
from torch.nn.utils import prune

import hermes_pruning as pruning
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A federated learning trainer of Hermes, used by both the client and the server."""

    def __init__(self, model=None, callbacks=None):
        """Initializes the global model."""
        super().__init__(model=model, callbacks=callbacks)
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

    def train_run_start(self, config):
        """Conducts pruning if needed before training."""
        super().train_run_start(config)

        # Evaluate if structured pruning should be conducted
        self.datasource = datasources_registry.get(client_id=self.client_id)
        self.testset = self.datasource.get_test_set()
        logging.info(
            "[Client #%d] Testing the model for prune decision.", self.client_id
        )
        accuracy = self.test_model(config, self.testset, None)
        self.pruned_amount = pruning.compute_pruned_amount(self.model, self.client_id)

        # Apply the most to the incoming server payload model to create the model for training
        self.model = self.apply_mask(self.model)

        # Send the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        logging.info(
            "[Client #%d] Evaluated Accuracy for pruning: %.2f%%",
            self.client_id,
            accuracy * 100,
        )

        if (
            accuracy >= self.accuracy_threshold
            and self.pruned_amount < self.pruning_target
        ):
            logging.info("[Client #%d] Conducting structured pruning.", self.client_id)

            if self.pruning_target - self.pruned_amount < self.pruning_rate:
                self.pruning_rate = (self.pruning_target - self.pruned_amount) / 100
                self.mask = pruning.structured_pruning(
                    self.model, self.pruning_rate, adjust_rate=self.pruned_amount
                )
            else:
                self.pruning_rate = (self.pruning_rate) / (100 - self.pruned_amount)
                self.mask = pruning.structured_pruning(
                    self.model,
                    self.pruning_rate,
                )

            self.save_mask(self.mask)
            self.need_prune = True
        else:
            logging.info("[Client #%d] No need to prune.", self.client_id)
            self.need_prune = False

    def train_run_end(self, config):
        """Method called at the end of training run."""
        # Make pruning permanent if it was conducted
        if self.need_prune or self.pruned_amount > 0:
            for __, layer in self.model.named_parameters():
                if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                    prune.remove(layer, "weight")

            self.pruned_amount = pruning.compute_pruned_amount(
                self.model, self.client_id
            )
            logging.info(
                "[Client #%d] Pruned Amount: %.2f%%", self.client_id, self.pruned_amount
            )

    def apply_mask(self, model):
        """Applies the mask onto the incoming personalized model."""
        model_name = Config().trainer.model_name
        model_path = Config().params["model_path"]

        mask_path = f"{model_path}/{model_name}_client{self.client_id}_mask.pth"
        if not os.path.exists(mask_path):
            return self.model
        else:
            with open(mask_path, "rb") as mask_file:
                mask = pickle.load(mask_file)

        return pruning.apply_mask(model, mask, self.device)

    def save_mask(self, mask):
        """Saves the mask for merging in future rounds if pruning has occured."""
        model_name = Config().trainer.model_name
        model_path = Config().params["model_path"]

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        mask_path = f"{model_path}/{model_name}_client{self.client_id}_mask.pth"

        with open(mask_path, "wb") as payload_file:
            pickle.dump(mask, payload_file)
