"""
The training loop that takes place on clients.
"""

import logging
import os

import pickle
import torch

import hermes_pruning as pruning
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A federated learning trainer of Hermes, used by both the client and the server."""

    def __init__(self, model=None):
        """Initializes the global model."""
        super().__init__(model=model)
        self.mask = None
        self.mask_for_merging = None
        self.pruning_target = Config().clients.pruning_target * 100
        self.pruned_amount = 0
        self.pruning_rate = Config().clients.pruning_amount * 100
        self.datasource = None
        self.testset = None
        self.need_prune = False
        self.extra_payload_path = None
        self.accuracy_threshold = (
            Config().clients.accuracy_threshold
            if hasattr(Config().clients, "accuracy_threshold")
            else 0.5
        )
        # Get the names of the layers to be pruned - torch.nn.Conv2d and torch.nn.Linear
        self.pruned_layer_names = []
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                tensor = module.weight.data
                conv_layer_name = [
                    name
                    for name in self.model.state_dict()
                    if torch.equal(tensor, self.model.state_dict()[name])
                ]
                self.pruned_layer_names.append(conv_layer_name[0])

    def train_run_start(self, config):
        """Conduct pruning if needed before training."""

        # Evaluate if structured pruning should be conducted
        self.datasource = datasources_registry.get(client_id=self.client_id)
        self.testset = self.datasource.get_test_set()
        accuracy = self.test_model(config, self.testset, None)
        self.pruned_amount = pruning.compute_pruned_amount(
            self.model, self.extra_payload_path
        )

        # Merge the incoming server payload model with the mask to create the model for training
        self.model = self.merge_model(self.model, self.pruned_layer_names)

        # Send the model to the device used for training
        self.model.to(self.device)
        self.model.train()

        logging.info(
            "[Client #%d] Evaluated Accuracy: %.2f%%", self.client_id, accuracy * 100
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

            self.save_mask()
            self.need_prune = True
        else:
            logging.info("[Client #%d] No need to prune.", self.client_id)
            self.need_prune = False

    def train_run_end(self, config):
        """Method called at the end of training run."""
        # The pruning is made permanent if it was conducted
        if self.need_prune or self.pruned_amount > 0:
            pruning.remove(self.model)

            self.pruned_amount = pruning.compute_pruned_amount(
                self.model, self.extra_payload_path
            )
            logging.info(
                "[Client #%d] Pruned Amount: %.2f%%", self.client_id, self.pruned_amount
            )

    def merge_model(self, model, pruned_layer_names):
        """Apply the mask onto the incoming personalized model."""
        if not os.path.exists(self.extra_payload_path):
            return self.model
        else:
            with open(self.extra_payload_path, "rb") as mask_file:
                mask = pickle.load(mask_file)

        return pruning.apply_mask(model, mask, self.device, pruned_layer_names)

    def from_numpy(self, tensor):
        """Converts a numpy array to a pytorch tensor"""
        return torch.from_numpy(tensor)

    def save_mask(self):
        """If pruning has occured, the mask is saved for merging in future rounds."""

        checkpoint_path = Config().params["checkpoint_path"]

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        with open(self.extra_payload_path, "wb") as payload_file:
            pickle.dump(self.mask, payload_file)
