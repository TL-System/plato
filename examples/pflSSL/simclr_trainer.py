"""
A personalized federated learning trainer using SimCLR.

"""

import torch

from plato.trainers import basic_personalized

from utils import MultiViewCollateWrapper


class Trainer(basic_personalized.Trainer):
    """A personalized federated learning trainer using the FedBABU algorithm."""

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
