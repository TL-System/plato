"""
The training and testing loops for PyTorch.
"""
import copy
import logging
import time

import torch

import subfedavg_pruning as pruning_processor
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.samplers import registry as samplers_registry
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A federated learning trainer for Sub-Fedavg algorithm."""

    def __init__(self, model=None, callbacks=None):
        """Initializes the trainer with the provided model."""
        super().__init__(model=model, callbacks=callbacks)
        self.mask = None
        self.pruning_target = (
            Config().clients.pruning_amount * 100
            if hasattr(Config().clients, "pruning_amount")
            else 40
        )
        self.pruning_amount = (
            Config().clients.pruning_amount * 100
            if hasattr(Config().clients, "pruning_amount")
            else 40
        )
        self.pruned = 0
        self.made_init_mask = False
        self.mask_distance_threshold = (
            Config().clients.mask_distance_threshold
            if hasattr(Config().clients, "mask_distance_threshold")
            else 0.0001
        )
        self.first_epoch_mask = None
        self.last_epoch_mask = None

        self.datasource = None
        self.testset = None
        self.testset_sampler = None
        self.testset_loaded = False
        self.accuracy_threshold = (
            Config().clients.accuracy_threshold
            if hasattr(Config().clients, "accuracy_threshold")
            else 0.5
        )

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Performs forward and backward passes in the training loop."""
        self.optimizer.zero_grad()

        outputs = self.model(examples)

        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        # Freeze pruned weights by zeroing their gradients
        step = 0
        for name, parameter in self.model.named_parameters():
            if "weight" in name:
                grad_tensor = parameter.grad.data.cpu().numpy()
                grad_tensor = grad_tensor * self.mask[step]
                parameter.grad.data = torch.from_numpy(grad_tensor).to(self.device)
                step = step + 1

        self.optimizer.step()

        return loss

    # pylint: disable=unused-argument
    def train_run_start(self, config):
        """Method called at the start of training run."""
        self.mask = pruning_processor.make_init_mask(self.model)

    def train_epoch_end(self, config):
        """Method called at the end of a training epoch."""
        if self.current_epoch == 1:
            self.first_epoch_mask = pruning_processor.fake_prune(
                self.pruning_amount,
                copy.deepcopy(self.model),
                copy.deepcopy(self.mask),
            )
        if self.current_epoch == config["epochs"]:
            self.last_epoch_mask = pruning_processor.fake_prune(
                self.pruning_amount,
                copy.deepcopy(self.model),
                copy.deepcopy(self.mask),
            )
        super().train_epoch_end(config)

    # pylint: disable=unused-argument
    def train_run_end(self, config):
        """Method called at the end of a training run."""
        self.process_pruning(self.first_epoch_mask, self.last_epoch_mask)

    def process_pruning(self, first_epoch_mask, last_epoch_mask):
        """Processes unstructed pruning."""
        mask_distance = pruning_processor.dist_masks(first_epoch_mask, last_epoch_mask)

        if (
            mask_distance > self.mask_distance_threshold
            and self.pruned < self.pruning_target
        ):
            if self.pruning_target - self.pruned < self.pruning_amount:
                self.pruning_amount = (
                    ((100 - self.pruned) - (100 - self.pruning_target))
                    / (100 - self.pruned)
                ) * 100
                self.pruning_amount = min(self.pruning_amount, 5)
                last_epoch_mask = pruning_processor.fake_prune(
                    self.pruning_amount,
                    copy.deepcopy(self.model),
                    copy.deepcopy(self.mask),
                )

            orginal_weights = copy.deepcopy(self.model.state_dict())
            pruned_weights = pruning_processor.real_prune(
                copy.deepcopy(self.model), last_epoch_mask
            )
            self.model.load_state_dict(pruned_weights, strict=True)

            logging.info(
                "[Client #%d] Evaluating if pruning should be conducted.",
                self.client_id,
            )
            accuracy = self.eval_test()
            if accuracy >= self.accuracy_threshold:
                logging.info("[Client #%d] Conducted pruning.", self.client_id)
                self.mask = copy.deepcopy(last_epoch_mask)
            else:
                logging.info("[Client #%d] No need to prune.", self.client_id)
                self.model.load_state_dict(orginal_weights, strict=True)

        self.pruned, _ = pruning_processor.compute_pruned_amount(self.model)

    def eval_test(self):
        """Tests if needs to update pruning mask and conduct pruning."""
        if not self.testset_loaded:
            self.datasource = datasources_registry.get(client_id=self.client_id)
            self.testset = self.datasource.get_test_set()
            if hasattr(Config().data, "testset_sampler"):
                # Set the sampler for test set
                self.testset_sampler = samplers_registry.get(
                    self.datasource, self.client_id, testing=True
                )
            self.testset_loaded = True

        self.model.eval()

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            if self.testset_sampler is None:
                test_loader = torch.utils.data.DataLoader(
                    self.testset, batch_size=Config().trainer.batch_size, shuffle=False
                )
            # Use a testing set following the same distribution as the training set
            else:
                test_loader = torch.utils.data.DataLoader(
                    self.testset,
                    batch_size=Config().trainer.batch_size,
                    shuffle=False,
                    sampler=self.testset_sampler,
                )

            correct = 0
            total = 0

            with torch.no_grad():
                for examples, labels in test_loader:
                    examples, labels = examples.to(self.device), labels.to(self.device)

                    outputs = self.model(examples)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total

        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        self.model.cpu()

        return accuracy
