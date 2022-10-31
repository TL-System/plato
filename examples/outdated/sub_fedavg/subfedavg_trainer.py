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

    def __init__(self, model=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model)
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

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """The default training loop when a custom training loop is not supplied."""
        batch_size = config["batch_size"]
        self.sampler = sampler
        tic = time.perf_counter()

        self.run_history.reset()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        self.train_loader = Trainer.get_train_loader(batch_size, trainset, sampler)

        # Initializing the loss criterion
        _loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        optimizer = self.get_optimizer(self.model)
        self.lr_scheduler = self.get_lr_scheduler(config, optimizer)
        optimizer = self._adjust_lr(config, self.lr_scheduler, optimizer)

        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]

        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            for batch_id, (examples, labels) in enumerate(self.train_loader):
                examples, labels = examples.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(examples)

                loss = _loss_criterion(outputs, labels)
                self._loss_tracker.update(loss, labels.size(0))

                if "create_graph" in config:
                    loss.backward(create_graph=config["create_graph"])
                else:
                    loss.backward()

                # Freezing Pruned weights by making their gradients Zero
                step = 0
                for name, parameter in self.model.named_parameters():
                    if "weight" in name:
                        grad_tensor = parameter.grad.data.cpu().numpy()
                        grad_tensor = grad_tensor * self.mask[step]
                        parameter.grad.data = torch.from_numpy(grad_tensor).to(
                            self.device
                        )
                        step = step + 1

                optimizer.step()

                self.train_step_end(config, batch=batch_id, loss=loss)
                self.callback_handler.call_event(
                    "on_train_step_end", self, config, batch=batch_id, loss=loss
                )

            self.lr_scheduler_step()

            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

            # Simulate client's speed
            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if (
                hasattr(Config().server, "request_update")
                and Config().server.request_update
            ):
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{self.current_epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            self.run_history.update_metric("train_loss", self._loss_tracker.average)
            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

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
        """Process unstructed pruning."""
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
        """Test if needs to update pruning mask and conduct pruning."""
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
