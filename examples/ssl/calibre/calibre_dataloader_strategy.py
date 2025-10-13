"""
DataLoader strategy for Calibre that handles SSL training and personalization phases.

During SSL training, the data loader uses a multi-view collate function that returns
SSLSamples (a list of views). During personalization, it uses standard data loading.
"""

from collections import UserList

import torch
import torch.utils.data as data
from lightly.data.multi_view_collate import MultiViewCollate

from plato.config import Config
from plato.trainers.strategies.base import DataLoaderStrategy, TrainingContext


class SSLSamples(UserList):
    """A container for SSL samples, which contains multiple views as a list."""

    def to(self, device):
        """Assign a list of views to the specific device."""
        for view_idx, view in enumerate(self.data):
            if isinstance(view, torch.Tensor):
                view = view.to(device)
                self[view_idx] = view

        return self  # Return self, not self.data


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


class CalibreDataLoaderStrategy(DataLoaderStrategy):
    """
    Data loader strategy for Calibre that handles both SSL training and personalization.

    During SSL training (rounds 1 to Config().trainer.rounds):
    - Uses MultiViewCollateWrapper to create multi-view samples
    - Returns SSLSamples containing multiple views

    During personalization (after Config().trainer.rounds):
    - Uses standard DataLoader with personalized dataset
    - Returns regular tensors
    """

    def create_train_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> data.DataLoader:
        """
        Create training data loader based on the current training phase.

        Args:
            trainset: Training dataset
            sampler: Data sampler
            batch_size: Batch size
            context: Training context with current round information

        Returns:
            Configured DataLoader for the current phase
        """
        current_round = context.current_round

        # Check if we're in the personalization phase
        if current_round > Config().trainer.rounds:
            return self._create_personalization_loader(
                trainset, sampler, batch_size, context
            )
        else:
            return self._create_ssl_loader(trainset, sampler, batch_size, context)

    def _create_ssl_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> data.DataLoader:
        """
        Create data loader for SSL training phase with multi-view collate.

        Args:
            trainset: Training dataset
            sampler: Data sampler (may be Plato Sampler object)
            batch_size: Batch size
            context: Training context

        Returns:
            DataLoader with MultiViewCollateWrapper
        """
        collate_fn = MultiViewCollateWrapper()

        # Handle Plato Sampler objects that have a get() method
        if hasattr(sampler, "get") and callable(sampler.get):
            sampler = sampler.get()

        return data.DataLoader(
            dataset=trainset,
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    def _create_personalization_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> data.DataLoader:
        """
        Create data loader for personalization phase.

        During personalization, we use the personalized_trainset stored in
        the trainer (if available) instead of the regular trainset.

        Args:
            trainset: Training dataset (may be overridden)
            sampler: Data sampler (may be Plato Sampler object)
            batch_size: Batch size
            context: Training context

        Returns:
            Standard DataLoader
        """
        # Get personalized trainset from context if available
        personalized_trainset = context.state.get("personalized_trainset")
        if personalized_trainset is not None:
            trainset = personalized_trainset

        # Handle Plato Sampler objects that have a get() method
        if hasattr(sampler, "get") and callable(sampler.get):
            sampler = sampler.get()

        return data.DataLoader(
            dataset=trainset,
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler,
        )
