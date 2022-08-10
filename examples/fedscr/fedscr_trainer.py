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
        self.use_adaptive = bool(
            hasattr(Config().clients, "adaptive") and Config().clients.adaptive
        )
        self.train_loss = None
        self.test_loss = None
        self.avg_update = None
        self.div_from_global = None
        self.orig_weights = None

    def prune_updates(self):
        """Prune the weight updates by setting some updates to 0."""
        self.load_all_grads()

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
                delta = trained_tensor - orig_tensor + self.all_grads[i]
                orig_delta = copy.deepcopy(delta)

                aggregated_channels = self.aggregate_channels(delta)
                aggregated_filters = self.aggregate_filters(delta)

                delta = self.prune_channel_updates(aggregated_channels, delta)
                delta = self.prune_filter_updates(aggregated_filters, delta)

                delta_name = f"{orig_name}.weight"
                self.all_grads[i] = orig_delta - delta
                conv_updates[delta_name] = delta
                i += 1

        for orig_key, trained_key in zip(
            self.orig_weights.state_dict(), self.model.state_dict()
        ):
            if not orig_key in conv_updates:
                orig_tensor = self.orig_weights.state_dict()[orig_key]
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

        for channel_index in range(num_channels):
            tensor_sum = 0
            for filters in range(num_filters):
                tensor_sum += np.abs(delta[filters, channel_index, :, :])
            aggregated_channels[channel_index] = tensor_sum

        for index, __ in enumerate(aggregated_channels):
            aggregated_channels[index] = np.sum(aggregated_channels[index])

        return aggregated_channels

    def aggregate_filters(self, delta):
        """Aggregate the sum of all channels from a single filter."""
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

    def prune_channel_updates(self, aggregated_channels, delta):
        """Prune the channel updates that lie below the FedSCR threshold."""
        for i, norm in enumerate(aggregated_channels):
            if norm < self.update_threshold:
                delta[:, i, :, :] = 0

        return delta

    def prune_filter_updates(self, aggregated_filters, delta):
        """Prune the filter updates that lie below the FedSCR threshold."""
        for i, norm in enumerate(aggregated_filters):
            if norm < self.update_threshold:
                delta[i, :, :, :] = 0

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
            tensor = self.total_grad[key].cpu()
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
        # Calculate weight divergence between local and global model, used for adaptive algorithm
        self.div_from_global = self.compute_weight_divergence()
        logging.info(
            "[Client #%d] Weight divergence: %.2f", self.client_id, self.div_from_global
        )

        checkpoint_path = Config().params["checkpoint_path"]
        model_name = Config().trainer.model_name

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Get the overall weight updates
        logging.info("[Client #%d] Pruning weight updates.", self.client_id)
        self.prune_updates()
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

        # Add weight divergence and average update to client report
        if self.use_adaptive is True:
            add_to_report = {
                "div_from_global": self.div_from_global,
                "avg_update": self.avg_update,
            }

            report_path = f"{checkpoint_path}/{model_name}_{self.client_id}.pkl"

            with open(report_path, "wb") as file:
                pickle.dump(add_to_report, file)

        # Save the final training loss
        model_name = config["model_name"]
        filename = f"{model_name}_{self.client_id}.loss"
        Trainer._save_loss(self.train_loss.data.item(), filename)

        self.model.load_state_dict(self.total_grad, strict=True)

    def compute_weight_divergence(self):
        """Calculate the divergence of the locally trained model from the global model."""
        self.orig_weights.to(self.device)
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

    def local_update_significance(self):
        """Calculate the average weight update."""
        delta = 0
        total = 0

        for key in sorted(self.total_grad.keys()):
            tensor = self.total_grad[key].cpu()
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
