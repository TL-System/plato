"""
The training loop that takes place on clients of FedSCR.
"""

import copy
import logging
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers.composable import ComposableTrainer


class FedSCRCallback(TrainerCallback):
    """Callback for FedSCR pruning and gradient accumulation operations."""

    def __init__(self):
        """Initialize the callback."""
        # The accumulated gradients for a client throughout the FL session
        self.acc_grads = []

    def on_train_run_start(self, trainer, config, **kwargs):
        """Initialize at the start of training."""
        trainer.total_grad = OrderedDict()
        trainer.orig_weights = copy.deepcopy(trainer.model)
        trainer.orig_weights.to(trainer.device)

    def on_train_run_end(self, trainer, config, **kwargs):
        """Process weight update pruning at the end of training."""
        # Get the overall weight updates
        logging.info("[Client #%d] Pruning weight updates.", trainer.client_id)
        trainer.prune_update(self.acc_grads)
        logging.info(
            "[Client #%d] SCR ratio (pruned amount): %.2f%%",
            trainer.client_id,
            trainer.compute_pruned_amount(),
        )

        # Add weight divergence and average update to client report
        if trainer.use_adaptive:
            # Calculate weight divergence between local and global model
            trainer.div_from_global = trainer.compute_weight_divergence()

            # Calculate average local weight updates
            trainer.avg_update = trainer.compute_local_update_significance()

            logging.info(
                "[Client #%d] Average local weight updates: %.2f",
                trainer.client_id,
                trainer.avg_update,
            )
            logging.info(
                "[Client #%d] Weight divergence: %.2f",
                trainer.client_id,
                trainer.div_from_global,
            )

            trainer.run_history.update_metric(
                "div_from_global", trainer.div_from_global
            )
            trainer.run_history.update_metric("avg_update", trainer.avg_update)

        trainer.model.load_state_dict(trainer.total_grad, strict=True)

        # Update accumulated gradients for next round
        self.acc_grads = trainer._acc_grads


class Trainer(ComposableTrainer):
    """A federated learning trainer used by the client."""

    def __init__(self, model=None, callbacks=None):
        """Initializes the trainer with the provided model."""
        # The threshold for determining whether an update is significant or not
        self.update_threshold = (
            Config().clients.update_threshold
            if hasattr(Config().clients, "update_threshold")
            else 0.3
        )

        # The overall weight updates applied to the model in a single round
        self.total_grad = OrderedDict()

        # The accumulated gradients for a client throughout the FL session
        self._acc_grads = []

        # Should the clients use the adaptive algorithm?
        self.use_adaptive = bool(
            hasattr(Config().clients, "adaptive") and Config().clients.adaptive
        )
        self.avg_update = None
        self.div_from_global = None
        self.orig_weights = None

        # Create callbacks for FedSCR
        fedscr_callbacks = [FedSCRCallback]
        if callbacks is not None:
            fedscr_callbacks.extend(callbacks)

        # Initialize with FedSCR callbacks
        super().__init__(
            model=model,
            callbacks=fedscr_callbacks,
        )

    def prune_update(self, acc_grads):
        """Prunes the weight update by setting some parameters in update to 0."""
        self._acc_grads = self.load_acc_grads()

        conv_updates = OrderedDict()
        i = 0
        for (orig_name, orig_module), (__, trained_module) in zip(
            self.orig_weights.named_modules(), self.model.named_modules()
        ):
            if isinstance(
                trained_module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
            ):
                orig_tensor = orig_module.weight.data.cpu().numpy()
                trained_tensor = trained_module.weight.data.cpu().numpy()
                delta = trained_tensor - orig_tensor + self._acc_grads[i]
                orig_delta = copy.deepcopy(delta)

                aggregated_channels = self.aggregate_channels(delta)
                aggregated_filters = self.aggregate_filters(delta)

                delta = self.prune_channels(aggregated_channels, delta)
                delta = self.prune_filters(aggregated_filters, delta)

                delta_name = f"{orig_name}.weight"
                self._acc_grads[i] = orig_delta - delta
                conv_updates[delta_name] = delta
                i += 1

        for orig_key, trained_key in zip(
            self.orig_weights.state_dict(), self.model.state_dict()
        ):
            if orig_key not in conv_updates:
                orig_tensor = self.orig_weights.state_dict()[orig_key]
                trained_tensor = self.model.state_dict()[trained_key]
                delta = trained_tensor - orig_tensor
                self.total_grad[orig_key] = delta
            else:
                self.total_grad[orig_key] = torch.from_numpy(conv_updates[orig_key])

        self.save_acc_grads()

    def aggregate_channels(self, delta):
        """Aggregates the sum of a certain channel from all filters."""
        num_channels = delta.shape[1]
        num_filters = delta.shape[0]
        aggregated_channels = [None] * num_channels

        for channel_index in range(num_channels):
            tensor_sum = 0
            for filters in range(num_filters):
                tensor_sum += np.abs(delta[filters, channel_index, :, :])
            aggregated_channels[channel_index] = tensor_sum

        for index, __ in enumerate(aggregated_channels):
            aggregated_channels[index] = np.sum(aggregated_channels[index])

        return aggregated_channels

    def aggregate_filters(self, delta):
        """Aggregates the sum of all channels from a single filter."""
        num_channels = delta.shape[1]
        num_filters = delta.shape[0]
        aggregated_filters = [None] * num_filters

        for filter_index in range(num_filters):
            tensor_sum = 0
            for channel in range(num_channels):
                tensor_sum += np.abs(delta[filter_index, channel, :, :])
            aggregated_filters[filter_index] = tensor_sum

        for index, __ in enumerate(aggregated_filters):
            aggregated_filters[index] = np.sum(aggregated_filters[index])

        return aggregated_filters

    def prune_channels(self, aggregated_channels, delta):
        """Prunes the channels in update that lie below the FedSCR threshold."""
        for i, norm in enumerate(aggregated_channels):
            if norm < self.update_threshold:
                delta[:, i, :, :] = 0

        return delta

    def prune_filters(self, aggregated_filters, delta):
        """Prunes the filters in update that lie below the FedSCR threshold."""
        for i, norm in enumerate(aggregated_filters):
            if norm < self.update_threshold:
                delta[i, :, :, :] = 0

        return delta

    def save_acc_grads(self):
        """Saves the accumulated client gradients for the next communication round."""
        model_name = Config().trainer.model_name
        checkpoint_path = Config().params["checkpoint_path"]

        acc_grad_path = (
            f"{checkpoint_path}/{model_name}_client{self.client_id}_grad.pth"
        )
        with open(acc_grad_path, "wb") as payload_file:
            pickle.dump(self._acc_grads, payload_file)

    def load_acc_grads(self):
        """Loads the accumulated gradients from a previous communication round."""
        model_name = Config().trainer.model_name
        checkpoint_path = Config().params["checkpoint_path"]

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        grad_path = f"{checkpoint_path}/{model_name}_client{self.client_id}_grad.pth"
        if os.path.exists(grad_path):
            with open(grad_path, "rb") as payload_file:
                return pickle.load(payload_file)
        else:
            count = 0
            for module in self.model.modules():
                if isinstance(
                    module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
                ):
                    count += 1
            return [0] * count

    def compute_pruned_amount(self):
        """Computes the pruned percentage of the entire model."""
        nonzero = 0
        total = 0
        for key in sorted(self.total_grad.keys()):
            tensor = self.total_grad[key].cpu()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params

        return 100 * (total - nonzero) / total

    def compute_weight_divergence(self):
        """Calculates the divergence of the locally trained model from the global model."""
        div_from_global = 0
        for (__, orig_module), (__, trained_module) in zip(
            self.orig_weights.named_modules(), self.model.named_modules()
        ):
            if isinstance(
                trained_module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
            ):
                orig_tensor = orig_module.weight.data.cpu()
                trained_tensor = trained_module.weight.data.cpu()
                div_from_global += (
                    torch.sum(torch.abs(trained_tensor - orig_tensor))
                    / torch.sum(torch.abs(trained_tensor))
                ).numpy()

        return np.sqrt(div_from_global)

    def compute_local_update_significance(self):
        """Calculates the average weight update."""
        delta = 0
        total = 0

        for key in sorted(self.total_grad.keys()):
            tensor = self.total_grad[key].cpu()
            delta += torch.sum(tensor).numpy()

        model = self.model.named_modules()
        for __, module in model:
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                tensor = module.weight.data.cpu()
                total += torch.sum(tensor).numpy()

        return np.sqrt(np.abs(delta / total))
