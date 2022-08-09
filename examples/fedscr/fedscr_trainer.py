"""
The training loop that takes place on clients of FedSCR.
"""
from collections import OrderedDict

import copy
import pickle
import os
import logging
import torch
import numpy as np

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A federated learning trainer used by the client."""

    def __init__(self, model=None):
        """Initialize the trainer with the provided model."""
        super().__init__(model=model)

        # The threshold for determining whether or not an update is significant
        self.update_threshold = (
            Config().clients.update_threshold
            if hasattr(Config().clients, "update_threshold")
            else 0.3
        )

        # The overall weight updates applied to the model in a single round.
        self.total_grad = OrderedDict()

        # The accumulated gradients for a client throughout the FL session.
        self.all_grads = []

        # Should the clients use the adaptive algorithm?
        self.use_adaptive = (
            True
            if hasattr(Config().clients, "adaptive") and Config().clients.adaptive
            else False
        )
        self.train_loss = None
        self.test_loss = None
        self.avg_update = None
        self.div = None

    def prune_updates(self, orig_weights):
        """Prune the weight updates by setting some updates to 0."""

        self.load_all_grads()

        coupled_models = zip(orig_weights.named_modules(), self.model.named_modules())

        conv_updates = OrderedDict()
        step = 0
        for (orig_name, orig_module), (__, trained_module) in coupled_models:
            if isinstance(
                trained_module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
            ):
                orig_tensor = orig_module.weight.data.cpu().numpy()
                trained_tensor = trained_module.weight.data.cpu().numpy()
                delta = trained_tensor - orig_tensor + self.all_grads[step]
                orig_delta = copy.deepcopy(delta)

                aggregated_channels = self.aggregate_channels(delta)
                aggregated_filters = self.aggregate_filters(delta)

                delta = self.prune_channel_updates(aggregated_channels, delta)
                delta = self.prune_filter_updates(aggregated_filters, delta)

                delta_name = f"{orig_name}.weight"
                self.all_grads[step] = orig_delta - delta
                conv_updates[delta_name] = delta
                step += 1

        coupled_models = zip(orig_weights.state_dict(), self.model.state_dict())
        for orig_key, trained_key in coupled_models:
            if not orig_key in conv_updates:
                orig_tensor = orig_weights.state_dict()[orig_key]
                trained_tensor = self.model.state_dict()[trained_key]
                delta = trained_tensor - orig_tensor
                self.total_grad[orig_key] = delta

            else:
                self.total_grad[orig_key] = torch.from_numpy(conv_updates[orig_key])

        self.save_gradient()

    def aggregate_channels(self, delta):
        """Aggregate the sum of a certain channel from all filters."""
        num_channels = delta.shape[1]
        num_filters = delta.shape[0]
        aggregated_channels = [None] * num_channels

        step = 0
        for channel in range(num_channels):
            tensor_sum = 0
            for filters in range(num_filters):
                tensor_sum += np.abs(delta[filters, channel, :, :])
            aggregated_channels[step] = tensor_sum
            step += 1

        for index, __ in enumerate(aggregated_channels):
            aggregated_channels[index] = np.sum(aggregated_channels[index])

        return aggregated_channels

    def aggregate_filters(self, delta):
        """Aggregate the sum of all channels from a single filter."""
        num_channels = delta.shape[1]
        num_filters = delta.shape[0]
        aggregated_filters = [None] * num_filters

        step = 0
        for filters in range(num_filters):
            tensor_sum = 0
            for channel in range(num_channels):
                tensor_sum += np.abs(delta[filters, channel, :, :])
            aggregated_filters[step] = tensor_sum
            step += 1

        for index, __ in enumerate(aggregated_filters):
            aggregated_filters[index] = np.sum(aggregated_filters[index])

        return aggregated_filters

    def prune_channel_updates(self, aggregated_channels, delta):
        """Prune the channel updates that lie below the FedSCR threshold."""
        for step, norm in enumerate(aggregated_channels):
            if norm < self.update_threshold:
                delta[:, step, :, :] = 0

        return delta

    def prune_filter_updates(self, aggregated_filters, delta):
        """Prune the filter updates that lie below the FedSCR threshold."""
        for step, norm in enumerate(aggregated_filters):
            if norm < self.update_threshold:
                delta[step, :, :, :] = 0

        return delta

    def save_gradient(self):
        """Save the client updated gradients for the next communication round."""
        model_name = Config().trainer.model_name
        checkpoint_path = Config().params["checkpoint_path"]

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        allgrad_path = f"{checkpoint_path}/{model_name}_client{self.client_id}_grad.pth"
        with open(allgrad_path, "wb") as payload_file:
            pickle.dump(self.all_grads, payload_file)

    def load_all_grads(self):
        """Load the gradients from a previous communication round."""
        model_name = Config().trainer.model_name
        checkpoint_path = Config().params["checkpoint_path"]

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        grad_path = f"{checkpoint_path}/{model_name}_client{self.client_id}_grad.pth"
        if os.path.exists(grad_path):
            with open(grad_path, "rb") as payload_file:
                self.all_grads = pickle.load(payload_file)
        else:
            count = 0
            for module in self.model.modules():
                if isinstance(
                    module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
                ):
                    count += 1
            self.all_grads = [0] * count

    def compute_pruned_amount(self):
        """
        This function computes the pruned percentage.
        :return pruned percentage, number of remaining weights:
        """
        nonzero = 0
        total = 0
        for key in sorted(self.total_grad.keys()):
            tensor = self.total_grad[key]
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params

        return 100 * (total - nonzero) / total

    def train_step_end(self, config, batch=None, loss=None):
        """Method called at the end of a training step."""
        self.train_loss = loss

    def train_run_start(self, config):
        """Method called at the start of training run."""
        self.total_grad = OrderedDict()
        self.orig_weights = copy.deepcopy(self.model)

    def train_run_end(self, config):
        """Method called at the end of training run."""
        # Calculate weight divergence between local and global model
        self.div = self.weight_div(self.orig_weights)
        logging.info("[Client #%d] Weight Divergence: %.2f", self.client_id, self.div)

        # Get the update threshold
        if self.use_adaptive:
            checkpoint_path = Config().params["checkpoint_path"]
            model_name = Config().trainer.model_name

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            div_path = f"{checkpoint_path}/{model_name}_thresholds.pkl"

            with open(div_path, "rb") as file:
                update_thresholds = pickle.load(file)

            self.update_threshold = update_thresholds[self.client_id - 1]
            logging.info(
                "[Client #%d] Update Threshold: %.2f",
                self.client_id,
                self.update_threshold,
            )

        # Get the overall weight updates as self.total_grad
        logging.info("[Client #%d] Pruning weight updates.", self.client_id)
        self.prune_updates(self.orig_weights)
        logging.info(
            "[Client #%d] SCR ratio (pruned amount): %.2f%%",
            self.client_id,
            self.compute_pruned_amount(),
        )

        # Calculate average local weight updates
        self.avg_update = self.local_update_significance()
        logging.info(
            "[Client #%d] Average local weight updates: %.2f",
            self.client_id,
            self.avg_update,
        )

        model_name = config["model_name"]
        filename = f"{model_name}_{self.client_id}.loss"
        Trainer._save_loss(self.train_loss.data.item(), filename)

    def weight_div(self, orig_weights):
        """Calculate the divergence of the locally trained model from the global model."""

        coupled_models = zip(orig_weights.named_modules(), self.model.named_modules())

        div = 0
        for (__, orig_module), (__, trained_module) in coupled_models:
            if isinstance(
                trained_module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
            ):
                orig_tensor = orig_module.weight.data.cpu()
                trained_tensor = trained_module.weight.data.cpu()
                div += (
                    torch.sum(torch.abs(trained_tensor - orig_tensor))
                    / torch.sum(torch.abs(trained_tensor))
                ).numpy()

        return np.sqrt(div)

    def local_update_significance(self):
        """Calculate the average weight update."""
        delta = 0
        total = 0

        for key in sorted(self.total_grad.keys()):
            tensor = self.total_grad[key]
            delta += torch.sum(tensor).numpy()

        model = self.model.named_modules()
        for (__, module) in model:
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                tensor = module.weight.data.cpu()
                total += torch.sum(tensor).numpy()

        return np.sqrt(np.abs(delta / total))

    @staticmethod
    def _save_loss(loss, filename):
        """Save the training loss to a file."""
        checkpoint_path = Config().params["checkpoint_path"]

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        loss_path = f"{checkpoint_path}/{filename}"
        with open(loss_path, "w", encoding="utf-8") as file:
            file.write(str(loss))

    @staticmethod
    def _load_loss(filename):
        """Load the training loss from a file."""
        checkpoint_path = Config().params["checkpoint_path"]
        loss_path = f"{checkpoint_path}/{filename}"

        with open(loss_path, "r", encoding="utf-8") as file:
            loss = float(file.read())

        return loss
