"""
A federated learning trainer using split learning.

Reference:

Vepakomma, et al., "Split learning for health: Distributed deep learning without sharing
raw patient data," in Proc. AI for Social Good Workshop, affiliated with ICLR 2018.

https://arxiv.org/pdf/1812.00564.pdf
"""
import logging
import os

import torch
from plato.config import Config

from plato.trainers import basic


class Trainer(basic.Trainer):
    """The split learning trainer."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        callbacks: The callbacks that this trainer uses.
        """
        super().__init__(model=model, callbacks=callbacks)
        self.cut_layer_grad = []

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """
        Creates an instance of the trainloader.

        Arguments:
        batch_size: the batch size.
        trainset: the training dataset.
        sampler: the sampler for the trainloader to use.
        """
        return trainset

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform the forward and backward passes of the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed.
        """
        examples = examples.detach().requires_grad_(True)

        loss = super().perform_forward_and_backward_passes(config, examples, labels)

        # Record gradients within the cut layer
        self.cut_layer_grad.append(examples.grad.clone().detach())

        return loss

    def train_run_end(self, config):
        """Saving recorded gradients to a file."""
        model_name = config["model_name"]
        model_path = Config().params["model_path"]

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_gradients_path = f"{model_path}/{model_name}_gradients.pth"
        torch.save(self.cut_layer_grad, model_gradients_path)

        logging.info(
            "[Server #%d] Gradients saved to %s.", os.getpid(), model_gradients_path
        )
