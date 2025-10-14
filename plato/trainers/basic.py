"""
The training and testing loops for PyTorch.

This module provides basic trainers using the composable trainer architecture.
The Trainer class uses the ComposableTrainer with default strategies, leveraging
the strategy design pattern.
"""

from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.lr_scheduler import TimmLRSchedulerStrategy


class Trainer(ComposableTrainer):
    """
    A basic federated learning trainer using the composable architecture.

    This trainer extends ComposableTrainer with default strategies.

    For advanced customization, use ComposableTrainer directly with custom strategies.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the basic trainer with default strategies.

        Arguments:
            model: The model to train (class or instance)
            callbacks: List of callback classes or instances
        """

        # Initialize with default strategies
        super().__init__(
            model=model,
            callbacks=None,
            loss_strategy=None,  # Uses DefaultLossCriterionStrategy
            optimizer_strategy=None,  # Uses DefaultOptimizerStrategy
            training_step_strategy=None,  # Uses DefaultTrainingStepStrategy
            lr_scheduler_strategy=None,  # Uses DefaultLRSchedulerStrategy
            model_update_strategy=None,  # Uses NoOpUpdateStrategy
            data_loader_strategy=None,  # Uses DefaultDataLoaderStrategy
            testing_strategy=None,  # Uses DefaultTestingStrategy
        )

        # Convenience attributes
        self._loss_criterion = None

    @property
    def loss_criterion(self):
        """Convenience property for accessing loss criterion."""
        if self._loss_criterion is None:
            # Create loss criterion using the strategy
            def compute_loss_fn(outputs, labels):
                return self.loss_strategy.compute_loss(outputs, labels, self.context)

            self._loss_criterion = compute_loss_fn
        return self._loss_criterion


class TimmSchedulerCallback(TrainerCallback):
    """
    Callback that handles timm scheduler-specific hooks.

    This callback calls the on_epoch_start() and on_step() methods
    on TimmLRSchedulerStrategy to handle timm's step_update() functionality.
    """

    def on_train_epoch_start(self, trainer, config, **kwargs):
        """Call timm scheduler's epoch start hook."""
        if isinstance(trainer.lr_scheduler_strategy, TimmLRSchedulerStrategy):
            trainer.lr_scheduler_strategy.on_epoch_start(
                trainer.lr_scheduler, trainer.context
            )

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        """Call timm scheduler's step hook after each training step."""
        if isinstance(trainer.lr_scheduler_strategy, TimmLRSchedulerStrategy):
            trainer.lr_scheduler_strategy.on_step(trainer.lr_scheduler, trainer.context)


class TrainerWithTimmScheduler(Trainer):
    """
    Trainer that works with timm schedulers using the composable architecture.

    This trainer uses a custom TimmLRSchedulerStrategy to handle timm's
    step_update() method that needs to be called after each training step.
    The timm-specific hooks are handled via TimmSchedulerCallback.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize trainer with timm scheduler strategy.

        Arguments:
            model: The model to train (class or instance)
            callbacks: List of callback classes or instances
        """
        # Create timm scheduler strategy
        timm_scheduler_strategy = TimmLRSchedulerStrategy()

        # Add TimmSchedulerCallback to support timm-specific callbacks
        callbacks_with_timm = [TimmSchedulerCallback]
        if callbacks is not None:
            callbacks_with_timm.extend(callbacks)

        # Initialize parent with timm strategy
        # We need to bypass Trainer.__init__ and call ComposableTrainer directly
        ComposableTrainer.__init__(
            self,
            model=model,
            callbacks=callbacks_with_timm,
            loss_strategy=None,
            optimizer_strategy=None,
            training_step_strategy=None,
            lr_scheduler_strategy=timm_scheduler_strategy,
            model_update_strategy=None,
            data_loader_strategy=None,
            testing_strategy=None,
        )

        # Convenience attributes
        self._loss_criterion = None
