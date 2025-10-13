"""
LR scheduler strategy for Calibre that handles SSL training and personalization phases.

During SSL training, uses the standard LR scheduler from config.
During personalization, uses a personalization-specific LR scheduler.
"""

from typing import Optional

import torch

from plato.config import Config
from plato.trainers import lr_schedulers
from plato.trainers.strategies.base import LRSchedulerStrategy, TrainingContext


class CalibreLRSchedulerStrategy(LRSchedulerStrategy):
    """
    LR scheduler strategy for Calibre's two-phase training.

    Phase 1 (SSL training): Uses standard LR scheduler from config
    Phase 2 (Personalization): Uses personalization-specific LR scheduler

    This strategy handles the case where the data loader length may not be
    available due to sampler limitations.
    """

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create LR scheduler based on the current training phase.

        Args:
            optimizer: The optimizer to schedule
            context: Training context with current round information

        Returns:
            LR scheduler instance, or None if no scheduling
        """
        current_round = context.current_round

        # Check if we're in the personalization phase
        if current_round > Config().trainer.rounds:
            return self._create_personalization_scheduler(optimizer, context)
        else:
            return self._create_ssl_scheduler(optimizer, context)

    def _create_ssl_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create LR scheduler for SSL training phase.

        Args:
            optimizer: The optimizer to schedule
            context: Training context

        Returns:
            LR scheduler instance, or None
        """
        # Get train_loader length safely
        train_loader_len = self._get_train_loader_length(context)

        if train_loader_len == 0:
            # Can't create scheduler without knowing loader length
            # Return None for no scheduling
            return None

        # Use framework's LR scheduler registry
        try:
            return lr_schedulers.get(optimizer, train_loader_len)
        except Exception:
            # If scheduler creation fails, return None
            return None

    def _create_personalization_scheduler(
        self, optimizer: torch.optim.Optimizer, context: TrainingContext
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create LR scheduler for personalization phase.

        Args:
            optimizer: The optimizer to schedule
            context: Training context

        Returns:
            LR scheduler instance, or None
        """
        # Check if personalization scheduler is configured
        if not hasattr(Config().algorithm, "personalization"):
            return None

        if not hasattr(Config().algorithm.personalization, "lr_scheduler"):
            return None

        # Get scheduler config
        lr_scheduler_name = Config().algorithm.personalization.lr_scheduler

        if hasattr(Config().parameters.personalization, "learning_rate"):
            lr_params = Config().parameters.personalization.learning_rate._asdict()
        else:
            lr_params = {}

        # Get train_loader length safely
        train_loader_len = self._get_train_loader_length(context)

        if train_loader_len == 0:
            # Can't create scheduler without knowing loader length
            return None

        try:
            return lr_schedulers.get(
                optimizer,
                train_loader_len,
                lr_scheduler=lr_scheduler_name,
                lr_params=lr_params,
            )
        except Exception:
            # If scheduler creation fails, return None
            return None

    def _get_train_loader_length(self, context: TrainingContext) -> int:
        """
        Safely get the length of the train loader.

        Args:
            context: Training context

        Returns:
            Length of train loader, or 0 if not available
        """
        train_loader = context.state.get("train_loader")

        if train_loader is None:
            return 0

        try:
            return len(train_loader)
        except (TypeError, AttributeError):
            # If len() fails (e.g., sampler doesn't support it),
            # try to estimate from dataset and batch size
            try:
                dataset_len = len(train_loader.dataset)
                batch_size = train_loader.batch_size or 1
                return (dataset_len + batch_size - 1) // batch_size
            except Exception:
                # If all else fails, return 0
                return 0
