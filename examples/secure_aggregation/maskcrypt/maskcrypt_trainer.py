"""
Customized trainer for MaskCrypt using the composable trainer architecture.

This module implements gradient computation for secure aggregation in federated
learning using the MaskCrypt protocol.
"""

import logging
import os
from typing import OrderedDict

import torch

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers.composable import ComposableTrainer


class GradientComputationCallback(TrainerCallback):
    """
    Callback for computing gradients on local data after training completes.

    This is used for secure aggregation protocols that require gradients
    rather than model weights.
    """

    def __init__(self):
        """Initialize the callback."""
        self.gradient = OrderedDict()

    def on_train_run_end(self, trainer, config, **kwargs):
        """Compute gradients on local data when training is finished."""
        logging.info(
            "[Client #%d] Training completed, computing gradient.",
            trainer.client_id,
        )

        # Set the existing gradients to zeros
        for param in trainer.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        trainer.model.to(trainer.device)
        trainer.model.train()

        # Get total number of samples
        if hasattr(trainer.sampler, "num_samples"):
            total_samples = trainer.sampler.num_samples()
        elif hasattr(trainer.sampler, "__len__"):
            total_samples = len(trainer.sampler)
        else:
            # If we can't get the sampler length, compute from dataset
            total_samples = len(trainer.train_loader.dataset)

        # Compute gradients over the entire training set
        for idx, (examples, labels) in enumerate(trainer.train_loader):
            examples, labels = examples.to(trainer.device), labels.to(trainer.device)
            outputs = trainer.model(examples)
            loss_criterion = torch.nn.CrossEntropyLoss()
            loss = loss_criterion(outputs, labels)
            # Weight the loss by batch proportion
            loss = loss * (len(labels) / total_samples)
            loss.backward()

        # Extract gradients from model parameters
        param_dict = dict(trainer.model.named_parameters())
        state_dict = trainer.model.state_dict()

        for name in state_dict.keys():
            if name in param_dict and param_dict[name].grad is not None:
                self.gradient[name] = param_dict[name].grad.clone()
            else:
                self.gradient[name] = torch.zeros(
                    state_dict[name].shape, device=trainer.device
                )

        # Save gradient to file
        model_type = config["model_name"]
        filename = f"{model_type}_gradient_{trainer.client_id}_{config['run_id']}.pth"
        save_gradient(self.gradient, filename)


def save_gradient(gradient, filename=None, location=None):
    """
    Save gradients to a file.

    Args:
        gradient: OrderedDict of gradients
        filename: Optional filename for the gradient file
        location: Optional directory location
    """
    model_path = Config().params["model_path"] if location is None else location
    model_name = Config().trainer.model_name

    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    except FileExistsError:
        pass

    if filename is not None:
        model_path = f"{model_path}/{filename}"
    else:
        model_path = f"{model_path}/{model_name}.pth"

    torch.save(gradient, model_path)


def load_gradient(filename=None, location=None):
    """
    Load gradients from a file.

    Args:
        filename: Optional filename for the gradient file
        location: Optional directory location

    Returns:
        OrderedDict of gradients
    """
    model_path = Config().params["model_path"] if location is None else location
    model_name = Config().trainer.model_name

    if filename is not None:
        model_path = f"{model_path}/{filename}"
    else:
        model_path = f"{model_path}/{model_name}.pth"

    return torch.load(model_path, weights_only=False)


class Trainer(ComposableTrainer):
    """
    MaskCrypt trainer with gradient computation for secure aggregation.

    This trainer extends ComposableTrainer and adds gradient computation
    functionality required by the MaskCrypt secure aggregation protocol.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the MaskCrypt trainer.

        Args:
            model: The model to train (class or instance)
            callbacks: List of callback classes or instances
        """
        # Create callbacks with gradient computation
        maskcrypt_callbacks = [GradientComputationCallback]
        if callbacks is not None:
            maskcrypt_callbacks.extend(callbacks)

        # Initialize with default strategies and MaskCrypt callbacks
        super().__init__(
            model=model,
            callbacks=maskcrypt_callbacks,
        )

        # Store reference to gradient computation callback
        self._gradient_callback = None
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, GradientComputationCallback):
                self._gradient_callback = callback
                break

    def get_gradient(self):
        """
        Read gradients from file and return to client.

        Returns:
            OrderedDict of gradients
        """
        model_type = Config().trainer.model_name
        run_id = Config().params["run_id"]
        filename = f"{model_type}_gradient_{self.client_id}_{run_id}.pth"

        return load_gradient(filename)

    @property
    def gradient(self):
        """
        Access computed gradients.

        Returns:
            OrderedDict of gradients from the gradient computation callback
        """
        if self._gradient_callback is not None:
            return self._gradient_callback.gradient
        return OrderedDict()

    @staticmethod
    def save_gradient(gradient, filename=None, location=None):
        """Save gradients to a file (static method for backward compatibility)."""
        return save_gradient(gradient, filename, location)

    @staticmethod
    def load_gradient(filename=None, location=None):
        """Load gradients from a file (static method for backward compatibility)."""
        return load_gradient(filename, location)
