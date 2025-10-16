"""
Data loader strategy implementations.

This module provides default and common data loader strategies for
the composable trainer architecture.
"""

import logging
from typing import Optional

import torch
import torch.utils.data

from plato.trainers.strategies.base import DataLoaderStrategy, TrainingContext


class DefaultDataLoaderStrategy(DataLoaderStrategy):
    """
    Default data loader strategy.

    Creates a standard PyTorch DataLoader with commonly used settings.

    Args:
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        shuffle: Whether to shuffle the data (usually False for FL with sampler)
        persistent_workers: Whether to keep workers alive between epochs

    Example:
        >>> strategy = DefaultDataLoaderStrategy(num_workers=4)
        >>> trainer = ComposableTrainer(data_loader_strategy=strategy)
    """

    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        shuffle: bool = False,
        persistent_workers: bool = False,
    ):
        """Initialize default data loader parameters."""
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.persistent_workers = persistent_workers

    def create_train_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> torch.utils.data.DataLoader:
        """Create standard training data loader."""
        # Handle different sampler types
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                # It's already a PyTorch Sampler object
                sampler_obj = sampler
                shuffle = False
            elif isinstance(sampler, (list, range)):
                # It's a list of indices, create SubsetRandomSampler
                sampler_obj = torch.utils.data.SubsetRandomSampler(sampler)
                shuffle = False
            elif hasattr(sampler, "get"):
                # It's a Plato Sampler, call get() to obtain PyTorch sampler
                sampler_obj = sampler.get()
                shuffle = False
            else:
                # Unknown type, try to use it directly
                sampler_obj = sampler
                shuffle = False
        else:
            sampler_obj = None
            shuffle = self.shuffle

        if sampler is None and not shuffle:
            logging.warning(
                "Data loader strategy received no sampler; falling back to SequentialSampler."
            )
        elif sampler is not None and sampler_obj is None:
            logging.warning(
                "Sampler %s did not provide indices; falling back to SequentialSampler.",
                type(sampler),
            )

        return torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler_obj,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
        )


class CustomCollateFnDataLoaderStrategy(DataLoaderStrategy):
    """
    Data loader strategy with custom collate function.

    Args:
        collate_fn: Custom collate function
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop the last incomplete batch

    Example:
        >>> def my_collate(batch):
        ...     # Custom collate logic
        ...     return batch
        >>> strategy = CustomCollateFnDataLoaderStrategy(collate_fn=my_collate)
        >>> trainer = ComposableTrainer(data_loader_strategy=strategy)
    """

    def __init__(
        self,
        collate_fn: callable,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        """Initialize custom collate data loader parameters."""
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def create_train_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> torch.utils.data.DataLoader:
        """Create data loader with custom collate function."""
        # Handle sampler
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                sampler_obj = sampler
            elif isinstance(sampler, (list, range)):
                sampler_obj = torch.utils.data.SubsetRandomSampler(sampler)
            elif hasattr(sampler, "get"):
                sampler_obj = sampler.get()
            else:
                sampler_obj = sampler
            shuffle = False
        else:
            sampler_obj = None
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler_obj,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )


class PrefetchDataLoaderStrategy(DataLoaderStrategy):
    """
    Data loader strategy with prefetching for faster data loading.

    This strategy creates multiple batches ahead of time to reduce
    waiting time between batches.

    Args:
        prefetch_factor: Number of batches to prefetch per worker
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop the last incomplete batch

    Example:
        >>> strategy = PrefetchDataLoaderStrategy(
        ...     prefetch_factor=4,
        ...     num_workers=4
        ... )
        >>> trainer = ComposableTrainer(data_loader_strategy=strategy)
    """

    def __init__(
        self,
        prefetch_factor: int = 2,
        num_workers: int = 2,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        """Initialize prefetch data loader parameters."""
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def create_train_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> torch.utils.data.DataLoader:
        """Create data loader with prefetching."""
        # Handle sampler
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                sampler_obj = sampler
            elif isinstance(sampler, (list, range)):
                sampler_obj = torch.utils.data.SubsetRandomSampler(sampler)
            elif hasattr(sampler, "get"):
                sampler_obj = sampler.get()
            else:
                sampler_obj = sampler
            shuffle = False
        else:
            sampler_obj = None
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler_obj,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class DynamicBatchSizeDataLoaderStrategy(DataLoaderStrategy):
    """
    Data loader strategy with dynamic batch size based on available memory.

    This strategy can adjust batch size based on GPU memory availability
    or other criteria.

    Args:
        initial_batch_size: Starting batch size
        max_batch_size: Maximum allowed batch size
        adjust_fn: Optional function to compute batch size based on context
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory

    Example:
        >>> def adjust_batch(context):
        ...     # Custom logic to determine batch size
        ...     return 32 if context.current_epoch < 10 else 64
        >>> strategy = DynamicBatchSizeDataLoaderStrategy(
        ...     initial_batch_size=32,
        ...     adjust_fn=adjust_batch
        ... )
        >>> trainer = ComposableTrainer(data_loader_strategy=strategy)
    """

    def __init__(
        self,
        initial_batch_size: int = 32,
        max_batch_size: int = 128,
        adjust_fn: Optional[callable] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        """Initialize dynamic batch size data loader parameters."""
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.adjust_fn = adjust_fn
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def create_train_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> torch.utils.data.DataLoader:
        """Create data loader with dynamic batch size."""
        # Determine actual batch size
        if self.adjust_fn is not None:
            actual_batch_size = self.adjust_fn(context)
            actual_batch_size = min(actual_batch_size, self.max_batch_size)
        else:
            actual_batch_size = batch_size

        # Handle sampler
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                sampler_obj = sampler
            elif isinstance(sampler, (list, range)):
                sampler_obj = torch.utils.data.SubsetRandomSampler(sampler)
            elif hasattr(sampler, "get"):
                sampler_obj = sampler.get()
            else:
                sampler_obj = sampler
            shuffle = False
        else:
            sampler_obj = None
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=actual_batch_size,
            shuffle=shuffle,
            sampler=sampler_obj,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class ShuffleDataLoaderStrategy(DataLoaderStrategy):
    """
    Data loader strategy that always shuffles data.

    This is useful when you want to ensure shuffling regardless of
    whether a sampler is provided.

    Args:
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop the last incomplete batch

    Example:
        >>> strategy = ShuffleDataLoaderStrategy(num_workers=2)
        >>> trainer = ComposableTrainer(data_loader_strategy=strategy)
    """

    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        """Initialize shuffle data loader parameters."""
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def create_train_loader(
        self, trainset, sampler, batch_size: int, context: TrainingContext
    ) -> torch.utils.data.DataLoader:
        """Create data loader with shuffling."""
        # If sampler is provided, convert to list and shuffle
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                sampler_obj = sampler
            elif isinstance(sampler, (list, range)):
                indices = list(sampler)
                sampler_obj = torch.utils.data.SubsetRandomSampler(indices)
            elif hasattr(sampler, "get"):
                sampler_obj = sampler.get()
            else:
                # Try to get indices from sampler
                try:
                    indices = list(sampler)
                    sampler_obj = torch.utils.data.SubsetRandomSampler(indices)
                except:
                    sampler_obj = sampler
            shuffle = False
        else:
            sampler_obj = None
            shuffle = True

        return torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler_obj,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
