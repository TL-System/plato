"""
The training and testing loops of PyTorch for personalized federated learning with
self-supervised learning.

"""

from typing import List, Tuple

import torch
from torch import Tensor
from lightly.data.multi_view_collate import MultiViewCollate

from plato.trainers import basic_personalized


class MultiViewCollateWrapper(MultiViewCollate):
    """An interface to connect the collate from lightly with the data loading schema of
    Plato."""

    def __call__(
        self, batch: List[Tuple[List[Tensor], int, str]]
    ) -> Tuple[List[Tensor], Tensor, List[str]]:
        views, labels, _ = super().__call__(batch)

        return views, labels


class Trainer(basic_personalized.Trainer):
    """A personalized federated learning trainer with self-supervised learning."""

    # pylint: disable=unused-argument
    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """
        Creates an instance of the trainloader.

        Arguments:
        batch_size: the batch size.
        trainset: the training dataset.
        sampler: the sampler for the trainloader to use.
        """
        collate_fn = MultiViewCollateWrapper()

        return torch.utils.data.DataLoader(
            dataset=trainset,
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
        )
