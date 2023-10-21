"""
A federated learning trainer using split learning.

Reference:

Vepakomma, et al., "Split Learning for Health: Distributed Deep Learning without Sharing
Raw Patient Data," in Proc. AI for Social Good Workshop, affiliated with ICLR 2018.

https://arxiv.org/pdf/1812.00564.pdf

Chopra, Ayush, et al. "AdaSplit: Adaptive Trade-offs for Resource-constrained Distributed
Deep Learning." arXiv preprint arXiv:2112.01637 (2021).

https://arxiv.org/pdf/2112.01637.pdf
"""

import logging
import os

import torch
from plato.config import Config

from plato.trainers import basic
from plato.datasources import feature
from plato.samplers import all_inclusive


# pylint:disable=too-many-instance-attributes
class Trainer(basic.Trainer):
    """The split learning trainer."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        callbacks: The callbacks that this trainer uses.
        """
        super().__init__(model=model, callbacks=callbacks)
        self.last_client_id = None
        self.last_optimizer = None

        # Client side variables
        self.training_samples = None
        self.gradients = None
        self.data_loader = None

        # Server side variables
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
        """Perform forward and backward passes in the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed.
        """
        if self.client_id == 0:
            return self._server_train_loop(config, examples, labels)

        return self._client_train_loop(examples)

    def train_run_end(self, config):
        """Additional tasks after training."""
        if self.client_id == 0:
            # Server needs to save gradients, clients not
            self.save_gradients(config)

    def get_optimizer(self, model):
        """Return the optimizer used in the last round to avoid reconfiguration."""
        if self.last_optimizer is None or self.last_client_id != self.client_id:
            self.last_optimizer = super().get_optimizer(model)
            self.last_client_id = self.client_id

        return self.last_optimizer

    def get_train_samples(self, batch_size, trainset, sampler):
        """
        Get a batch of training samples to extract feature, the trainer has to save these
        samples to complete training later.
        """
        data_loader = torch.utils.data.DataLoader(
            dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
        )
        data_loader = iter(data_loader)
        self.training_samples = next(data_loader)
        return self.training_samples

    def retrieve_train_samples(self):
        """Retrieve the training samples to complete client training."""
        # Wrap the training samples with datasource and sampler to be fed into Plato trainer
        self.training_samples = self.process_training_samples_before_retreiving(
            self.training_samples
        )

        samples = feature.DataSource([[self.training_samples]])
        sampler = all_inclusive.Sampler(samples)

        return samples, sampler

    def load_gradients(self, gradients):
        """Load the gradients which will be used to complete client training."""
        self.gradients = gradients

    def _client_train_loop(self, examples):
        """Complete the client side training with gradients from server."""
        self.optimizer.zero_grad()
        examples = self.process_samples_before_client_forwarding(examples)
        outputs = self.model.forward_to(examples)

        # Backpropagate with gradients from the server
        outputs.backward(self.gradients)
        self.optimizer.step()

        # No loss value on the client side
        loss = torch.zeros(1)
        self._loss_tracker.update(loss, examples.size(0))
        return loss

    def _server_train_loop(self, config, examples, labels):
        """The training loop on the server."""
        self.optimizer.zero_grad()
        loss, grad, batch_size = self.server_forward_from((examples, labels), config)
        loss = loss.cpu().detach()
        self._loss_tracker.update(loss, batch_size)

        # Record gradients within the cut layer
        if grad is not None:
            grad = grad.cpu().clone().detach()
        self.cut_layer_grad = [grad]
        self.optimizer.step()

        logging.warning(
            "[Server #%d] Gradients computed with training loss: %.4f",
            os.getpid(),
            loss,
        )

        return loss

    def save_gradients(self, config):
        """Server saves recorded gradients to a file."""
        model_name = config["model_name"]
        model_path = Config().params["model_path"]

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_gradients_path = f"{model_path}/{model_name}_gradients.pth"
        torch.save(self.cut_layer_grad, model_gradients_path)

        logging.info(
            "[Server #%d] Gradients saved to %s.", os.getpid(), model_gradients_path
        )

    def get_gradients(self):
        """Read gradients from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        model_gradients_path = f"{model_path}/{model_name}_gradients.pth"
        logging.info(
            "[Server #%d] Loading gradients from %s.", os.getpid(), model_gradients_path
        )

        return torch.load(model_gradients_path)

    # API functions for split learning
    def process_training_samples_before_retreiving(self, training_samples):
        """Process training samples before completing retreiving samples."""
        return training_samples

    def process_samples_before_client_forwarding(self, examples):
        """Process the examples befor client conducting forwarding."""
        return examples

    # pylint:disable=unused-argument
    def server_forward_from(self, batch, config):
        """
        The event for server completing training by forwarding from intermediate features.
        Uses may override this function for training different models with split learning.

        Inputs:
            batch: the batch of inputs for forwarding.
            config: training configuration.
        Returns:
            loss: the calculated loss.
            grad: the gradients over the intermediate feature.
            batch_size: the batch size of the current sample.
        """

        inputs, target = batch
        batch_size = inputs.size(0)
        inputs = inputs.detach().requires_grad_(True)
        outputs = self.model.forward_from(inputs)
        loss = self._loss_criterion(outputs, target)
        loss.backward()
        grad = inputs.grad
        return loss, grad, batch_size
