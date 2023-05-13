"""
The training and testing loops of PyTorch for personalized federated learning with
self-supervised learning.

"""

from typing import List, Tuple
from warnings import warn
from collections import UserList

import torch
from torch import Tensor
from lightly.data.multi_view_collate import MultiViewCollate
from tqdm import tqdm

from plato.trainers import loss_criterion
from plato.config import Config

from bases import personalized_trainer


class ExamplesList(UserList):
    """The list containing multiple examples."""

    def to(self, device):
        """Assign the tensor item into the specific device."""
        for example_idx, example in enumerate(self.data):
            if hasattr(example, "to"):
                if isinstance(example, torch.Tensor):
                    example = example.to(device)
                else:
                    example.to(device)
                self.__setitem__(example_idx, example)

        return self.data


class MultiViewCollateWrapper(MultiViewCollate):
    """An interface to connect the collate from lightly with the data loading schema of
    Plato."""

    def __call__(
        self, batch: List[Tuple[List[Tensor], int, str]]
    ) -> Tuple[List[Tensor], Tensor, List[str]]:
        """Turns a batch of tuples into single tuple."""
        if len(batch) == 0:
            warn("MultiViewCollate received empty batch.")
            return [], [], []

        views = ExamplesList([[] for _ in range(len(batch[0][0]))])
        labels = []
        fnames = []
        for sample in batch:
            img, label = sample[0], sample[1]
            fname = sample[3] if len(sample) == 3 else None
            for i, view in enumerate(img):
                views[i].append(view.unsqueeze(0))
            labels.append(label)
            if fname is not None:
                fnames.append(fname)

        for i, view in enumerate(views):
            views[i] = torch.cat(view)

        labels = torch.tensor(
            labels, dtype=torch.long
        )  # Conversion to tensor to ensure backwards compatibility

        if fnames:  # Compatible with lightly
            return views, labels, fnames
        else:  # Compatible with Plato
            return views, labels


class Trainer(personalized_trainer.Trainer):
    """A personalized federated learning trainer with self-supervised learning."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model, callbacks=callbacks)

        # dataset for personalization
        self.personalized_trainset = None
        self.personalized_testset = None

        # By default, if `personalized_sampler` is not set up, it will
        # be equal to the `sampler`.
        self.personalized_sampler = None
        self.personalized_testset_sampler = None

    def set_personalized_trainset(self, dataset):
        """set the testset."""
        self.personalized_trainset = dataset

    def set_personalized_trainset_sampler(self, dataset):
        """set the sampler for personalized trainset."""
        self.personalized_sampler = dataset

    def set_personalized_testset(self, dataset):
        """set the testset."""
        self.personalized_testset = dataset

    def set_personalized_testset_sampler(self, sampler):
        """set the sampler for the testset."""
        self.personalized_testset_sampler = sampler

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """Obtain the training loader based on the learning mode."""
        if self.personalized_learning:
            personalized_config = Config().algorithm.personalization._asdict()
            batch_size = personalized_config["batch_size"]
            trainset = self.personalized_trainset
            sampler = self.personalized_sampler

            return torch.utils.data.DataLoader(
                dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
            )
        else:
            collate_fn = MultiViewCollateWrapper()

            return torch.utils.data.DataLoader(
                dataset=trainset,
                shuffle=False,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
            )

    # pylint: disable=unused-argument
    def get_test_loader(self, batch_size, **kwargs):
        """Getting one test loader based on the learning mode."""
        testset = self.testset
        sampler = self.testset_sampler

        if self.personalized_learning:
            testset = self.personalized_testset
            sampler = self.personalized_testset_sampler

        return torch.utils.data.DataLoader(
            dataset=testset,
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler,
        )

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        if not self.personalized_learning:
            defined_ssl_loss = loss_criterion.get()

            def compute_plato_loss(outputs, labels):
                if isinstance(outputs, (list, tuple)):
                    return defined_ssl_loss(*outputs)
                else:
                    return defined_ssl_loss(outputs)

            return compute_plato_loss

        loss_criterion_type = Config().algorithm.personalization.loss_criterion
        loss_criterion_params = (
            Config().parameters.personalization.loss_criterion._asdict()
        )
        return loss_criterion.get(
            loss_criterion=loss_criterion_type,
            loss_criterion_params=loss_criterion_params,
        )

    def preprocess_personalized_model(self, config):
        """Do nothing to the loaded personalized mdoel."""

    def personalized_model_forward(self, examples):
        """Forward the input examples to the personalized model."""

        # Extract representation from the trained
        # frozen encoder of ssl.
        # No optimization is reuqired by this encoder.
        with torch.no_grad():
            features = self.model.encoder(examples)

        # Perfrom the training and compute the loss
        return self.personalized_model(features)
