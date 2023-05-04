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

from plato.trainers import basic_personalized
from plato.trainers import loss_criterion


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

    def get_loss_criterion(self):
        """Returns the loss criterion.
        As the loss functions derive from the lightly,
        it is desired to create a interface
        """

        defined_ssl_loss = loss_criterion.get()

        def compute_plato_loss(outputs, labels):
            if isinstance(outputs, (list, tuple)):
                return defined_ssl_loss(*outputs)
            else:
                return defined_ssl_loss(outputs)

        return compute_plato_loss

    def perform_evaluation(self, data_loader, defined_model=None, **kwargs):
        """The operation of performing the evaluation on the testset with the defined model."""
        # Define the test phase of the eval stage
        defined_model = (
            self.personalized_model if defined_model is None else defined_model
        )
        defined_model.eval()
        defined_model.to(self.device)
        self.model.to(self.device)

        correct = 0
        total = 0

        with torch.no_grad():
            for _, (examples, labels) in enumerate(data_loader):
                examples, labels = examples.to(self.device), labels.to(self.device)

                features = self.model.encoder(examples)
                outputs = defined_model(features)

                outputs = self.process_personalized_outputs(outputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        eval_outputs = {"accuracy": accuracy}

        return eval_outputs

    def personalized_train_one_epoch(
        self,
        epoch,
        config,
        epoch_loss_meter,
    ):
        # pylint:disable=too-many-arguments
        """Performing one epoch of learning for the personalization."""

        epoch_loss_meter.reset()
        self.personalized_model.train()
        self.personalized_model.to(self.device)
        self.model.to(self.device)

        pers_epochs = config["personalized_epochs"]

        local_progress = tqdm(
            self.personalized_train_loader,
            desc=f"Epoch {epoch}/{pers_epochs+1}",
            disable=True,
        )

        for _, (examples, labels) in enumerate(local_progress):
            examples, labels = examples.to(self.device), labels.to(self.device)
            # Clear the previous gradient
            self.personalized_optimizer.zero_grad()

            # Extract representation from the trained
            # frozen encoder of ssl.
            # No optimization is reuqired by this encoder.
            with torch.no_grad():
                features = self.model.encoder(examples)

            # Perfrom the training and compute the loss
            preds = self.personalized_model(features)
            loss = self._personalized_loss_criterion(preds, labels)

            # Perfrom the optimization
            loss.backward()
            self.personalized_optimizer.step()

            # Update the epoch loss container
            epoch_loss_meter.update(loss, labels.size(0))

            local_progress.set_postfix(
                {
                    "lr": self.personalized_lr_scheduler,
                    "loss": epoch_loss_meter.loss_value,
                    "loss_avg": epoch_loss_meter.average,
                }
            )

        return epoch_loss_meter
