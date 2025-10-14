"""
The training and testing loops for PyTorch.
"""

import copy
import logging

import subfedavg_pruning as pruning_processor
import torch

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.samplers import registry as samplers_registry
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import TrainingStepStrategy


class SubFedAvgTrainingStepStrategy(TrainingStepStrategy):
    """Training step strategy that applies gradient masking for pruning."""

    def __init__(self, trainer):
        """Initialize with reference to trainer for accessing mask."""
        self.trainer = trainer

    def training_step(
        self, model, optimizer, examples, labels, loss_criterion, context
    ):
        """Perform training step with gradient masking."""
        optimizer.zero_grad()
        outputs = model(examples)
        loss = loss_criterion(outputs, labels)

        # Handle create_graph config for second-order gradients
        if "create_graph" in context.config:
            loss.backward(create_graph=context.config["create_graph"])
        else:
            loss.backward()

        # Freeze pruned weights by zeroing their gradients
        step = 0
        for name, parameter in model.named_parameters():
            if "weight" in name:
                grad_tensor = parameter.grad.data.cpu().numpy()
                grad_tensor = grad_tensor * self.trainer.mask[step]
                parameter.grad.data = torch.from_numpy(grad_tensor).to(context.device)
                step = step + 1

        optimizer.step()

        return loss


class PruningCallback(TrainerCallback):
    """Callback for Sub-FedAvg pruning operations."""

    def __init__(self):
        """Initialize the callback."""
        self.first_epoch_mask = None
        self.last_epoch_mask = None
        self.datasource = None
        self.testset = None
        self.testset_sampler = None
        self.testset_loaded = False

    def on_train_run_start(self, trainer, config, **kwargs):
        """Initialize pruning mask at the start of training."""
        trainer.mask = pruning_processor.make_init_mask(trainer.model)

    def on_train_epoch_end(self, trainer, config, **kwargs):
        """Track masks at first and last epochs."""
        if trainer.current_epoch == 1:
            self.first_epoch_mask = pruning_processor.fake_prune(
                trainer.pruning_amount,
                copy.deepcopy(trainer.model),
                copy.deepcopy(trainer.mask),
            )
        if trainer.current_epoch == config["epochs"]:
            self.last_epoch_mask = pruning_processor.fake_prune(
                trainer.pruning_amount,
                copy.deepcopy(trainer.model),
                copy.deepcopy(trainer.mask),
            )

    def on_train_run_end(self, trainer, config, **kwargs):
        """Process pruning at the end of training."""
        self.process_pruning(trainer, self.first_epoch_mask, self.last_epoch_mask)

    def process_pruning(self, trainer, first_epoch_mask, last_epoch_mask):
        """Processes unstructured pruning."""
        mask_distance = pruning_processor.dist_masks(first_epoch_mask, last_epoch_mask)

        if (
            mask_distance > trainer.mask_distance_threshold
            and trainer.pruned < trainer.pruning_target
        ):
            if trainer.pruning_target - trainer.pruned < trainer.pruning_amount:
                trainer.pruning_amount = (
                    ((100 - trainer.pruned) - (100 - trainer.pruning_target))
                    / (100 - trainer.pruned)
                ) * 100
                trainer.pruning_amount = min(trainer.pruning_amount, 5)
                last_epoch_mask = pruning_processor.fake_prune(
                    trainer.pruning_amount,
                    copy.deepcopy(trainer.model),
                    copy.deepcopy(trainer.mask),
                )

            original_weights = copy.deepcopy(trainer.model.state_dict())
            pruned_weights = pruning_processor.real_prune(
                copy.deepcopy(trainer.model), last_epoch_mask
            )
            trainer.model.load_state_dict(pruned_weights, strict=True)

            logging.info(
                "[Client #%d] Evaluating if pruning should be conducted.",
                trainer.client_id,
            )
            accuracy = self.eval_test(trainer)
            if accuracy >= trainer.accuracy_threshold:
                logging.info("[Client #%d] Conducted pruning.", trainer.client_id)
                trainer.mask = copy.deepcopy(last_epoch_mask)
            else:
                logging.info("[Client #%d] No need to prune.", trainer.client_id)
                trainer.model.load_state_dict(original_weights, strict=True)

        trainer.pruned, _ = pruning_processor.compute_pruned_amount(trainer.model)

    def eval_test(self, trainer):
        """Tests if needs to update pruning mask and conduct pruning."""
        if not self.testset_loaded:
            self.datasource = datasources_registry.get(client_id=trainer.client_id)
            self.testset = self.datasource.get_test_set()
            if hasattr(Config().data, "testset_sampler"):
                # Set the sampler for test set
                self.testset_sampler = samplers_registry.get(
                    self.datasource, trainer.client_id, testing=True
                )
            self.testset_loaded = True

        trainer.model.eval()

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
                # Handle different sampler types properly
                if isinstance(self.testset_sampler, torch.utils.data.Sampler):
                    sampler_obj = self.testset_sampler
                elif isinstance(self.testset_sampler, (list, range)):
                    sampler_obj = torch.utils.data.SubsetRandomSampler(
                        self.testset_sampler
                    )
                elif hasattr(self.testset_sampler, "get"):
                    sampler_obj = self.testset_sampler.get()
                else:
                    sampler_obj = self.testset_sampler

                test_loader = torch.utils.data.DataLoader(
                    self.testset,
                    batch_size=Config().trainer.batch_size,
                    shuffle=False,
                    sampler=sampler_obj,
                )

            correct = 0
            total = 0

            with torch.no_grad():
                for examples, labels in test_loader:
                    examples, labels = (
                        examples.to(trainer.device),
                        labels.to(trainer.device),
                    )

                    outputs = trainer.model(examples)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total

        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", trainer.client_id)
            raise testing_exception

        trainer.model.cpu()

        return accuracy


class Trainer(ComposableTrainer):
    """A federated learning trainer for Sub-FedAvg algorithm."""

    def __init__(self, model=None, callbacks=None):
        """Initializes the trainer with the provided model."""
        # Initialize pruning parameters
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
        self.mask_distance_threshold = (
            Config().clients.mask_distance_threshold
            if hasattr(Config().clients, "mask_distance_threshold")
            else 0.0001
        )
        self.accuracy_threshold = (
            Config().clients.accuracy_threshold
            if hasattr(Config().clients, "accuracy_threshold")
            else 0.5
        )

        # Create training step strategy with gradient masking
        training_step_strategy = SubFedAvgTrainingStepStrategy(self)

        # Create callbacks for pruning
        pruning_callbacks = [PruningCallback]
        if callbacks is not None:
            pruning_callbacks.extend(callbacks)

        # Initialize with Sub-FedAvg strategies and callbacks
        super().__init__(
            model=model,
            callbacks=pruning_callbacks,
            training_step_strategy=training_step_strategy,
        )
