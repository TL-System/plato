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
        # Wrap the training samples with datasource and sampler to be fed into Plato trainer
        self.training_samples = self.process_training_samples_before_retrieving(
            self.training_samples
        )
        return self.training_samples

    def retrieve_train_samples(self):
        """Retrieve the training samples to complete client training."""
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
        gradients = self.gradients
        gradients[0] = gradients[0].to(self.device)
        outputs.backward(gradients)
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

        if "/" in model_name:
            model_name = model_name.replace("/", "_")

        model_gradients_path = f"{model_path}/{model_name}_gradients.pth"
        torch.save(self.cut_layer_grad, model_gradients_path)

        logging.info(
            "[Server #%d] Gradients saved to %s.", os.getpid(), model_gradients_path
        )

    def get_gradients(self):
        """Read gradients from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if "/" in model_name:
            model_name = model_name.replace("/", "_")

        model_gradients_path = f"{model_path}/{model_name}_gradients.pth"
        logging.info(
            "[Server #%d] Loading gradients from %s.", os.getpid(), model_gradients_path
        )

        return torch.load(model_gradients_path)

    def test_model(self, config, testset, sampler=None, **kwargs):
        """
        Evaluates the model with the provided test dataset and test sampler.

        Arguments:
        testset: the test dataset.
        sampler: the test sampler. The default is None.
        kwargs (optional): Additional keyword arguments.
        """
        batch_size = config["batch_size"]
        accuracy = self.test_model_split_learning(batch_size, testset, sampler)
        return accuracy

    # API functions for split learning
    def process_training_samples_before_retrieving(self, training_samples) -> ...:
        """Process training samples before completing retrieving samples."""
        return training_samples

    def process_samples_before_client_forwarding(self, examples) -> ...:
        """Process the examples before client conducting forwarding."""
        return examples

    # pylint:disable=unused-argument
    def server_forward_from(self, batch, config) -> (..., ..., int):
        """
        The event for server completing training by forwarding from intermediate features.
        Users may override this function for training different models with split learning.

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

    def update_weights_before_cut(self, current_weights, weights) -> ...:
        """
        Update the weights before cut layer, called when testing accuracy in trainer.
        Inputs:
        current_weights: the current weights extracted by the algorithm.
        weights: the weights to load.
        Output:
        current_weights: the updated current weights of the model.
        """
        cut_layer_idx = self.model.layers.index(self.model.cut_layer)

        for i in range(0, cut_layer_idx):
            weight_name = f"{self.model.layers[i]}.weight"
            bias_name = f"{self.model.layers[i]}.bias"

            if weight_name in current_weights:
                current_weights[weight_name] = weights[weight_name]

            if bias_name in current_weights:
                current_weights[bias_name] = weights[bias_name]

        return current_weights

    def forward_to_intermediate_feature(self, inputs, targets) -> (..., ...):
        """
        The process to forward to get intermediate feature on the client.
        Arguments:
        inputs: the inputs for the model on the clients.
        targets: the targets to get of the whole model.

        Return:
        outputs: the intermediate feature.
        targets: the targets to get of the whole model.
        """
        with torch.no_grad():
            logits = self.model.forward_to(inputs)

        outputs = logits.detach().cpu()
        targets = targets.detach().cpu()
        return outputs, targets

    def test_model_split_learning(self, batch_size, testset, sampler=None) -> ...:
        """
        The test model process for split learning.

        Returns:
        accuracy: the metrics for evaluating the model.
        """
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler
        )
        correct = 0
        total = 0

        self.model.to(self.device)
        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(self.device)

                outputs = self.model(examples)

                outputs = self.process_outputs(outputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
