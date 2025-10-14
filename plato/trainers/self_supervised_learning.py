"""
A self-supervised learning (SSL) trainer for SSL training and testing.

Federated learning with SSL trains the global model based on the data loader and
objective function of SSL algorithms. For this unsupervised learning process, we
cannot test the model directly as the model only extracts features from the
data. Therefore, we use KNN as a classifier to get the accuracy of the global
model during the regular federated training process.

In the personalization process, each client trains a linear layer locally, based
on the features extracted by the trained global model.

The accuracy obtained by KNN during the regular federated training rounds may
not be used to compare with the accuracy in supervised learning methods.
"""

import logging
from collections import UserList

import torch
from lightly.data.multi_view_collate import MultiViewCollate

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import loss_criterion, lr_schedulers, optimizers
from plato.trainers.basic import Trainer as BasicTrainer
from plato.trainers.strategies.base import (
    DataLoaderStrategy,
    LossCriterionStrategy,
    LRSchedulerStrategy,
    OptimizerStrategy,
    TestingStrategy,
    TrainingContext,
    TrainingStepStrategy,
)


class SSLSamples(UserList):
    """A container for SSL sample, which contains multiple views as a list."""

    def to(self, device):
        """Assign a list of views into the specific device."""
        for view_idx, view in enumerate(self.data):
            if isinstance(view, torch.Tensor):
                view = view.to(device)

            self[view_idx] = view

        return self.data


class MultiViewCollateWrapper(MultiViewCollate):
    """
    An interface to connect collate from lightly with Plato's data loading mechanism.
    """

    def __call__(self, batch):
        """Turn a batch of tuples into a single tuple."""
        # Add a fname to each sample to make the batch compatible with lightly
        batch = [batch[i] + (" ",) for i in range(len(batch))]

        # Process first two parts with the lightly collate
        views, labels, _ = super().__call__(batch)

        # Assign views, which is a list of tensors, into SSLSamples
        samples = SSLSamples(views)
        return samples, labels


# ============================================================================
# Custom Strategies for SSL Training
# ============================================================================


class SSLDataLoaderStrategy(DataLoaderStrategy):
    """
    Data loader strategy for SSL training with dual-phase support.

    Uses MultiViewCollate during SSL training phase and standard loader
    during personalization phase.
    """

    def __init__(self, personalized_trainset=None):
        """Initialize the SSL data loader strategy."""
        self.personalized_trainset = personalized_trainset

    def create_train_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> torch.utils.data.DataLoader:
        """Create data loader based on training phase (SSL or personalization)."""
        current_round = context.current_round

        # Handle different sampler types
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                # It's already a PyTorch Sampler object
                sampler_obj = sampler
            elif isinstance(sampler, (list, range)):
                # It's a list of indices, create SubsetRandomSampler
                sampler_obj = torch.utils.data.SubsetRandomSampler(sampler)
            elif hasattr(sampler, "get"):
                # It's a Plato Sampler, call get() to obtain PyTorch sampler
                sampler_obj = sampler.get()
            else:
                # Unknown type, try to use it directly
                sampler_obj = sampler
        else:
            sampler_obj = None

        # Personalization phase: use simple data loader
        if current_round > Config().trainer.rounds:
            dataset = (
                self.personalized_trainset if self.personalized_trainset else trainset
            )
            return torch.utils.data.DataLoader(
                dataset=dataset,
                shuffle=False,
                batch_size=batch_size,
                sampler=sampler_obj,
            )
        # SSL training phase: use multi-view collate
        else:
            collate_fn = MultiViewCollateWrapper()
            return torch.utils.data.DataLoader(
                dataset=trainset,
                shuffle=False,
                batch_size=batch_size,
                sampler=sampler_obj,
                collate_fn=collate_fn,
            )


class SSLLossCriterionStrategy(LossCriterionStrategy):
    """
    Loss criterion strategy for SSL with dual-phase support.

    Uses SSL-specific loss during training phase and classification loss
    during personalization phase.
    """

    def __init__(self):
        """Initialize the SSL loss strategy."""
        self._ssl_criterion = None
        self._personalization_criterion = None

    def setup(self, context: TrainingContext) -> None:
        """Initialize loss criteria for both phases."""
        # SSL loss criterion - store directly without wrapper
        self._ssl_criterion = loss_criterion.get()

    def compute_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, context: TrainingContext
    ) -> torch.Tensor:
        """Compute loss based on current training phase."""
        current_round = context.current_round

        # Personalization phase
        if current_round > Config().trainer.rounds:
            if self._personalization_criterion is None:
                loss_criterion_type = Config().algorithm.personalization.loss_criterion
                loss_criterion_params = {}
                if hasattr(Config().parameters.personalization, "loss_criterion"):
                    loss_criterion_params = (
                        Config().parameters.personalization.loss_criterion._asdict()
                    )
                self._personalization_criterion = loss_criterion.get(
                    loss_criterion=loss_criterion_type,
                    loss_criterion_params=loss_criterion_params,
                )
            return self._personalization_criterion(outputs, labels)
        # SSL training phase
        else:
            # Handle different output types
            if isinstance(outputs, (list, tuple)):
                return self._ssl_criterion(*outputs)
            return self._ssl_criterion(outputs)


class SSLOptimizerStrategy(OptimizerStrategy):
    """
    Optimizer strategy for SSL with dual-phase support.

    Uses different optimizers for SSL training and personalization phases.
    """

    def __init__(self, local_layers=None):
        """Initialize the SSL optimizer strategy."""
        self.local_layers = local_layers

    def create_optimizer(
        self, model, context: TrainingContext
    ) -> torch.optim.Optimizer:
        """Create optimizer based on current training phase."""
        current_round = context.current_round

        # Personalization phase: optimize local layers only
        if current_round > Config().trainer.rounds:
            optimizer_name = Config().algorithm.personalization.optimizer
            optimizer_params = Config().parameters.personalization.optimizer._asdict()
            return optimizers.get(
                self.local_layers,
                optimizer_name=optimizer_name,
                optimizer_params=optimizer_params,
            )
        # SSL training phase: optimize full model
        else:
            return optimizers.get(model)


class SSLLRSchedulerStrategy(LRSchedulerStrategy):
    """
    LR scheduler strategy for SSL with dual-phase support.
    """

    def __init__(self):
        """Initialize the SSL LR scheduler strategy."""
        pass

    def _get_train_loader_length(self, train_loader):
        """
        Safely calculate the length of the train loader.

        Args:
            train_loader: PyTorch DataLoader instance

        Returns:
            int: Number of batches in the loader, or 0 if cannot be determined
        """
        if train_loader is None:
            return 0

        try:
            # Try direct length calculation
            return len(train_loader)
        except TypeError:
            # If sampler doesn't implement __len__, calculate from dataset
            try:
                dataset = train_loader.dataset
                batch_size = train_loader.batch_size
                if hasattr(dataset, "__len__") and batch_size:
                    dataset_size = len(dataset)
                    # Calculate number of batches (ceiling division)
                    return (dataset_size + batch_size - 1) // batch_size
            except (AttributeError, TypeError):
                pass

        # If all else fails, return 0
        return 0

    def create_scheduler(self, optimizer, context: TrainingContext):
        """Create LR scheduler based on current training phase."""
        current_round = context.current_round

        # Personalization phase
        if current_round > Config().trainer.rounds:
            lr_scheduler_name = Config().algorithm.personalization.lr_scheduler
            lr_params = Config().parameters.personalization.learning_rate._asdict()
            train_loader = context.state.get("train_loader")
            num_batches = self._get_train_loader_length(train_loader)

            return lr_schedulers.get(
                optimizer,
                num_batches,
                lr_scheduler=lr_scheduler_name,
                lr_params=lr_params,
            )
        # SSL training phase
        else:
            config = Config().trainer._asdict()
            # Remove 'optimizer' from config to avoid conflict with positional argument
            config.pop("optimizer", None)
            train_loader = context.state.get("train_loader")
            num_batches = self._get_train_loader_length(train_loader)
            return lr_schedulers.get(optimizer, num_batches, **config)


class SSLTrainingStepStrategy(TrainingStepStrategy):
    """
    Training step strategy for SSL with dual-phase support.

    During SSL training: trains full model
    During personalization: freezes encoder and trains only local layers
    """

    def __init__(self, local_layers=None):
        """Initialize the SSL training step strategy."""
        self.local_layers = local_layers

    def training_step(
        self,
        model,
        optimizer,
        examples,
        labels,
        loss_criterion: callable,
        context: TrainingContext,
    ) -> torch.Tensor:
        """Perform training step based on current phase."""
        current_round = context.current_round

        # Personalization phase: use frozen encoder + local layers
        if current_round > Config().trainer.rounds:
            optimizer.zero_grad()

            # Extract features using frozen encoder
            features = model.encoder(examples)
            # Train local layers
            outputs = self.local_layers(features)

            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            return loss
        # SSL training phase: standard training
        else:
            optimizer.zero_grad()
            outputs = model(examples)
            loss = loss_criterion(outputs, labels)

            # Support for create_graph in config
            if "create_graph" in context.config and context.config["create_graph"]:
                loss.backward(create_graph=True)
            else:
                loss.backward()

            optimizer.step()
            return loss


# ============================================================================
# Custom Testing Strategy for SSL
# ============================================================================


class SSLTestingStrategy(TestingStrategy):
    """
    Testing strategy for SSL with dual-phase support.

    During SSL training phase: Uses KNN classifier for evaluation
    During personalization phase: Uses trained local layers for evaluation
    """

    def __init__(self, local_layers=None, personalized_trainset=None):
        """
        Initialize SSL testing strategy.

        Args:
            local_layers: The personalized local layers (for personalization phase)
            personalized_trainset: Training set for personalization (used for KNN)
        """
        self.local_layers = local_layers
        self.personalized_trainset = personalized_trainset

    def _convert_sampler(self, sampler):
        """
        Convert Plato sampler to PyTorch sampler.

        Args:
            sampler: Plato Sampler, PyTorch Sampler, list of indices, or None

        Returns:
            PyTorch Sampler object or None
        """
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                # It's already a PyTorch Sampler object
                return sampler
            elif isinstance(sampler, (list, range)):
                # It's a list of indices, create SubsetRandomSampler
                return torch.utils.data.SubsetRandomSampler(sampler)
            elif hasattr(sampler, "get"):
                # It's a Plato Sampler, call get() to obtain PyTorch sampler
                return sampler.get()
            else:
                # Unknown type, try to use it directly
                return sampler
        return None

    def test_model(self, model, config, testset, sampler, context):
        """
        Test the model using KNN for SSL or local layers for personalization.

        Args:
            model: The model to test
            config: Testing configuration
            testset: Test dataset
            sampler: Optional data sampler
            context: Training context

        Returns:
            Test accuracy as float
        """
        batch_size = config["batch_size"]
        current_round = context.current_round

        # Personalization phase: test with local layers
        if current_round > Config().trainer.rounds:
            return self._test_with_local_layers(
                model, testset, sampler, batch_size, context
            )
        # SSL training phase: test with KNN
        else:
            return self._test_with_knn(model, testset, sampler, batch_size, context)

    def _test_with_local_layers(self, model, testset, sampler, batch_size, context):
        """Test using trained local layers."""
        self.local_layers.eval()
        self.local_layers.to(context.device)

        model.eval()
        model.to(context.device)

        # Convert sampler to PyTorch sampler
        sampler_obj = self._convert_sampler(sampler)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler_obj
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = (
                    examples.to(context.device),
                    labels.to(context.device),
                )

                features = model.encoder(examples)
                outputs = self.local_layers(features)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    def _test_with_knn(self, model, testset, sampler, batch_size, context):
        """Test using KNN classifier on extracted features."""
        logging.info("[Client #%d] Testing the model with KNN.", context.client_id)

        # Convert sampler to PyTorch sampler
        sampler_obj = self._convert_sampler(sampler)

        # Get the training loader and test loader
        train_loader = torch.utils.data.DataLoader(
            dataset=self.personalized_trainset,
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler_obj,
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler_obj
        )

        # Collect encodings
        train_encodings, train_labels = self._collect_encodings(
            model, train_loader, context
        )
        test_encodings, test_labels = self._collect_encodings(
            model, test_loader, context
        )

        # Build KNN and perform prediction
        distances = torch.cdist(test_encodings, train_encodings, p=2)
        knn = distances.topk(1, largest=False)
        nearest_idx = knn.indices
        predicted_labels = train_labels[nearest_idx].view(-1)
        test_labels = test_labels.view(-1)

        # Compute accuracy
        num_correct = torch.sum(predicted_labels == test_labels).item()
        accuracy = num_correct / len(test_labels)

        return accuracy

    def _collect_encodings(self, model, data_loader, context):
        """Collect encodings of the data by using the model encoder."""
        samples_encoding = None
        samples_label = None
        model.eval()
        model.to(context.device)

        for examples, labels in data_loader:
            examples, labels = examples.to(context.device), labels.to(context.device)
            with torch.no_grad():
                features = model.encoder(examples)
                if samples_encoding is None:
                    samples_encoding = features
                else:
                    samples_encoding = torch.cat([samples_encoding, features], dim=0)
                if samples_label is None:
                    samples_label = labels
                else:
                    samples_label = torch.cat([samples_label, labels], dim=0)

        return samples_encoding, samples_label


# ============================================================================
# Custom Callbacks for SSL Training
# ============================================================================


class SSLTrainRunStartCallback(TrainerCallback):
    """
    Callback to handle train run start for SSL trainer.

    Configures batch size and epochs for personalization phase,
    and moves local layers to device.
    """

    def __init__(self, local_layers=None):
        """Initialize with reference to local layers."""
        self.local_layers = local_layers

    def on_train_run_start(self, trainer, config, **kwargs):
        """Configure training parameters based on phase."""
        if trainer.current_round > Config().trainer.rounds:
            # Set config for personalization
            config["batch_size"] = Config().algorithm.personalization.batch_size
            config["epochs"] = Config().algorithm.personalization.epochs

            # Move local layers to device and set to train mode
            if self.local_layers is not None:
                self.local_layers.to(trainer.device)
                self.local_layers.train()


# ============================================================================
# SSL Trainer
# ============================================================================


class Trainer(BasicTrainer):
    """A federated SSL trainer using composable architecture."""

    def __init__(self, model=None, callbacks=None):
        """Initialize the SSL trainer with custom strategies."""
        # Datasets for personalization
        self.personalized_trainset = None
        self.personalized_testset = None

        # Initialize model first if needed to access encoder
        if model is None:
            temp_model = models_registry.get()
        elif callable(model):
            temp_model = model()
        else:
            temp_model = model

        # Define the personalized local layers
        model_params = Config().parameters.personalization.model._asdict()
        model_params["input_dim"] = temp_model.encoder.encoding_dim
        model_params["output_dim"] = model_params["num_classes"]
        self.local_layers = models_registry.get(
            model_name=Config().algorithm.personalization.model_name,
            model_type=Config().algorithm.personalization.model_type,
            model_params=model_params,
        )

        # Create custom strategies for SSL
        ssl_data_loader_strategy = SSLDataLoaderStrategy(
            personalized_trainset=self.personalized_trainset
        )
        ssl_loss_strategy = SSLLossCriterionStrategy()
        ssl_optimizer_strategy = SSLOptimizerStrategy(local_layers=self.local_layers)
        ssl_lr_scheduler_strategy = SSLLRSchedulerStrategy()
        ssl_training_step_strategy = SSLTrainingStepStrategy(
            local_layers=self.local_layers
        )
        ssl_testing_strategy = SSLTestingStrategy(
            local_layers=self.local_layers,
            personalized_trainset=self.personalized_trainset,
        )

        # Create custom callback for train run start
        ssl_callback = SSLTrainRunStartCallback(local_layers=self.local_layers)

        # Combine with provided callbacks
        all_callbacks = [ssl_callback]
        if callbacks is not None:
            all_callbacks.extend(callbacks)

        # Initialize parent with model and custom strategies
        # Note: We bypass BasicTrainer.__init__ to directly use ComposableTrainer
        from plato.callbacks.handler import CallbackHandler
        from plato.callbacks.trainer import LogProgressCallback
        from plato.trainers import tracking
        from plato.trainers.basic import LegacyHookBridgeCallback
        from plato.trainers.composable import ComposableTrainer

        # Initialize ComposableTrainer
        ComposableTrainer.__init__(
            self,
            model=temp_model,
            callbacks=None,  # We'll set up callbacks manually
            loss_strategy=ssl_loss_strategy,
            optimizer_strategy=ssl_optimizer_strategy,
            training_step_strategy=ssl_training_step_strategy,
            lr_scheduler_strategy=ssl_lr_scheduler_strategy,
            model_update_strategy=None,
            data_loader_strategy=ssl_data_loader_strategy,
            testing_strategy=ssl_testing_strategy,
        )

        # Setup callbacks with both legacy bridge and SSL callbacks
        self.callbacks = (
            [LegacyHookBridgeCallback] + all_callbacks + [LogProgressCallback]
        )
        self.callback_handler = CallbackHandler(self.callbacks)

        # Reinitialize tracking (parent might have set these up already)
        self.run_history = tracking.RunHistory()
        self._loss_tracker = tracking.LossTracker()

        # Convenience attributes
        self._loss_criterion = None

    def set_personalized_datasets(self, trainset, testset):
        """Set the personalized datasets for both training and testing."""
        self.personalized_trainset = trainset
        self.personalized_testset = testset

        # Update the data loader strategy with new personalized trainset
        if hasattr(self, "data_loader_strategy"):
            self.data_loader_strategy.personalized_trainset = trainset

        # Update the testing strategy with new personalized trainset
        if hasattr(self, "testing_strategy"):
            self.testing_strategy.personalized_trainset = trainset
