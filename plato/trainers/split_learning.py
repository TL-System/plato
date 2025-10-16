"""
A federated learning trainer using split learning with composable architecture.

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

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.datasources import feature
from plato.samplers import all_inclusive
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.base import (
    TestingStrategy,
    TrainingContext,
    TrainingStepStrategy,
)


class SplitLearningCallback(TrainerCallback):
    """
    Callback to handle split learning-specific tasks.

    This callback:
    - Injects trainer reference into context at training start
    - Saves gradients at the end of training (server only)
    """

    def on_train_run_start(self, trainer, config, **kwargs):
        """Inject trainer reference into context for strategy access."""
        trainer.context.state["trainer"] = trainer

    def on_train_run_end(self, trainer, config, **kwargs):
        """Save gradients after training (server only)."""
        if trainer.client_id == 0:
            # Server needs to save gradients, clients not
            trainer.save_gradients(config)
            logging.info("[Server #%d] Gradients saved after training.", os.getpid())


class SplitLearningTrainingStepStrategy(TrainingStepStrategy):
    """
    Training step strategy for split learning.

    This strategy implements the split learning protocol:
    - Client: Forward to cut layer, backprop with gradients from server
    - Server: Forward from cut layer, compute loss and gradients
    """

    def __init__(self):
        super().__init__()
        self.cut_layer_grad = []

    def training_step(
        self,
        model,
        optimizer,
        examples,
        labels,
        loss_criterion,
        context: TrainingContext,
    ):
        """
        Perform one split learning training step.

        Args:
            model: The model to train
            optimizer: The optimizer
            examples: Input batch (already moved to device)
            labels: Target labels (already moved to device)
            loss_criterion: Loss computation function
            context: Training context

        Returns:
            Loss value for this step
        """
        # Get trainer reference from context
        trainer = context.state.get("trainer")
        if trainer is None:
            raise ValueError("Trainer must be stored in context.state['trainer']")

        # Different behavior for server (client_id=0) vs clients
        if context.client_id == 0:
            return self._server_train_step(
                model, optimizer, examples, labels, loss_criterion, context, trainer
            )
        else:
            return self._client_train_step(
                model, optimizer, examples, labels, loss_criterion, context, trainer
            )

    def _client_train_step(
        self, model, optimizer, examples, labels, loss_criterion, context, trainer
    ):
        """Complete the client side training with gradients from server."""
        optimizer.zero_grad()

        examples, batch_size = trainer.process_samples_before_client_forwarding(
            examples
        )
        outputs = model.forward_to(examples)

        # Backpropagate with gradients from the server
        gradients = trainer.gradients[0] if trainer.gradients else None
        if gradients is None:
            logging.warning("[Client #%d] Gradients from server is None.", os.getpid())
        else:
            gradients = gradients.to(context.device)
            outputs.backward(gradients)
            optimizer.step()

        # No loss value on the client side
        loss = torch.zeros(1)
        return loss

    def _server_train_step(
        self, model, optimizer, examples, labels, loss_criterion, context, trainer
    ):
        """The training loop on the server."""
        optimizer.zero_grad()

        config = context.config
        loss, grad, batch_size = trainer.server_forward_from((examples, labels), config)
        loss = loss.cpu().detach()

        # Record gradients within the cut layer
        if grad is not None:
            grad = grad.cpu().clone().detach()
        self.cut_layer_grad = [grad]
        trainer.cut_layer_grad = self.cut_layer_grad

        optimizer.step()

        logging.warning(
            "[Server #%d] Gradients computed with training loss: %.4f",
            os.getpid(),
            loss,
        )

        return loss


class SplitLearningTestingStrategy(TestingStrategy):
    """
    Testing strategy for split learning models.

    This strategy implements the split learning test protocol where
    the model is tested as a whole (not split across client/server).
    """

    def test_model(self, model, config, testset, sampler, context):
        """
        Test the split learning model.

        Args:
            model: The model to test
            config: Testing configuration dictionary
            testset: Test dataset
            sampler: Optional data sampler for test set
            context: Training context

        Returns:
            Test accuracy as float
        """
        # Get trainer reference from context
        trainer = context.state.get("trainer")
        if trainer is None:
            raise ValueError("Trainer must be stored in context.state['trainer']")

        batch_size = config["batch_size"]
        sampler_obj = None
        if sampler is not None:
            if hasattr(sampler, "get"):
                sampler_obj = sampler.get()
            else:
                sampler_obj = sampler

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler_obj
        )

        correct = 0
        total = 0

        model.to(context.device)
        model.eval()

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = (
                    examples.to(context.device),
                    labels.to(context.device),
                )
                outputs = model(examples)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy


# pylint:disable=too-many-instance-attributes
class Trainer(ComposableTrainer):
    """
    The split learning trainer using composable architecture.

    This trainer uses ComposableTrainer with custom strategies and callbacks
    to implement split learning without inheritance.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize split learning trainer with custom strategies.

        Arguments:
            model: The model to train (class or instance)
            callbacks: Additional callback classes or instances
        """
        # Create split learning-specific strategies
        split_learning_training_strategy = SplitLearningTrainingStepStrategy()
        split_learning_testing_strategy = SplitLearningTestingStrategy()
        split_learning_callback = SplitLearningCallback()

        # Combine with user callbacks
        callbacks_with_split = [split_learning_callback]
        if callbacks is not None:
            callbacks_with_split.extend(callbacks)

        # Initialize with split learning strategies
        super().__init__(
            model=model,
            callbacks=callbacks_with_split,
            loss_strategy=None,  # Uses DefaultLossCriterionStrategy
            optimizer_strategy=None,  # Uses DefaultOptimizerStrategy
            training_step_strategy=split_learning_training_strategy,
            lr_scheduler_strategy=None,  # Uses DefaultLRSchedulerStrategy
            model_update_strategy=None,  # Uses NoOpUpdateStrategy
            data_loader_strategy=None,  # Uses DefaultDataLoaderStrategy
            testing_strategy=split_learning_testing_strategy,
        )

        # Split learning-specific attributes
        # Client side variables
        self.training_samples = None
        self.gradients = None
        self.data_loader = None

        # Server side variables
        self.cut_layer_grad = []

    def get_train_samples(self, batch_size, trainset, sampler):
        """
        Get a batch of training samples to extract feature.

        The trainer has to save these samples to complete training later.

        Arguments:
            batch_size: Batch size
            trainset: Training dataset
            sampler: Data sampler

        Returns:
            Training samples
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
        """
        Retrieve the training samples to complete client training.

        Returns:
            Tuple of (datasource, sampler)
        """
        samples = feature.DataSource([[self.training_samples]])
        sampler = all_inclusive.Sampler(samples)

        return samples, sampler

    def load_gradients(self, gradients):
        """
        Load the gradients which will be used to complete client training.

        Arguments:
            gradients: Gradients from server
        """
        self.gradients = gradients

    def save_gradients(self, config):
        """
        Server saves recorded gradients to a file.

        Arguments:
            config: Training configuration
        """
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
        """
        Read gradients from a file.

        Returns:
            Loaded gradients
        """
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if "/" in model_name:
            model_name = model_name.replace("/", "_")

        model_gradients_path = f"{model_path}/{model_name}_gradients.pth"
        logging.info(
            "[Server #%d] Loading gradients from %s.", os.getpid(), model_gradients_path
        )

        return torch.load(model_gradients_path)

    # API functions for split learning - can be overridden by subclasses

    def process_training_samples_before_retrieving(self, training_samples):
        """
        Process training samples before completing retrieving samples.

        Override this in subclasses for custom preprocessing.

        Arguments:
            training_samples: Raw training samples

        Returns:
            Processed training samples
        """
        return training_samples

    def process_samples_before_client_forwarding(self, examples):
        """
        Process the examples before client conducting forwarding.

        Override this in subclasses for custom preprocessing.

        Arguments:
            examples: Input examples

        Returns:
            Tuple of (processed_examples, batch_size)
        """
        return examples, examples.size(0)

    # pylint:disable=unused-argument
    def server_forward_from(self, batch, config):
        """
        The event for server completing training by forwarding from intermediate features.

        Users may override this function for training different models with split learning.

        Arguments:
            batch: the batch of inputs for forwarding
            config: training configuration

        Returns:
            Tuple of (loss, grad, batch_size):
                - loss: the calculated loss
                - grad: the gradients over the intermediate feature
                - batch_size: the batch size of the current sample
        """
        inputs, target = batch
        batch_size = inputs.size(0)
        inputs = inputs.detach().requires_grad_(True)
        outputs = self.model.forward_from(inputs)

        # Get loss criterion from strategy
        loss = self.loss_strategy.compute_loss(outputs, target, self.context)
        loss.backward()
        grad = inputs.grad

        return loss, grad, batch_size

    def update_weights_before_cut(self, current_weights, weights):
        """
        Update the weights before cut layer.

        Called when testing accuracy in trainer.

        Arguments:
            current_weights: the current weights extracted by the algorithm
            weights: the weights to load

        Returns:
            Updated current weights of the model
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

    def forward_to_intermediate_feature(self, inputs, targets):
        """
        The process to forward to get intermediate feature on the client.

        Arguments:
            inputs: the inputs for the model on the clients
            targets: the targets to get of the whole model

        Returns:
            Tuple of (outputs, targets):
                - outputs: the intermediate feature
                - targets: the targets to get of the whole model
        """
        with torch.no_grad():
            logits = self.model.forward_to(inputs)

        outputs = logits.detach().cpu()
        targets = targets.detach().cpu()
        return outputs, targets
