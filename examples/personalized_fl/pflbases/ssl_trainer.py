"""
A base trainer to perform training and testing loops for self-supervised learning.
"""

import logging
from collections import UserList

import torch
from lightly.data.multi_view_collate import MultiViewCollate

from plato.trainers import loss_criterion
from plato.config import Config
from plato.trainers import basic
from plato.models import registry as models_registry


class SSLSamples(UserList):
    """A SSL sample."""

    def to(self, device):
        """Assign the tensor item into the specific device."""
        for example_idx, example in enumerate(self.data):
            if hasattr(example, "to"):
                if isinstance(example, torch.Tensor):
                    example = example.to(device)
                else:
                    example.to(device)
                self[example_idx] = example

        return self.data


class MultiViewCollateWrapper(MultiViewCollate):
    """An interface to connect the collate from lightly with the data loading schema of
    Plato."""

    def __call__(self, batch):
        """Turns a batch of tuples into single tuple."""

        views = SSLSamples([[] for _ in range(len(batch[0][0]))])
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
        )  # Conversion to tensor to ensure backwards compatibility.

        # Compatible with lightly
        if fnames:
            return views, labels, fnames
        # Compatible with Plato.
        return views, labels


class Trainer(basic.Trainer):
    """A personalized federated learning trainer with self-supervised learning."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model, callbacks=callbacks)

        # Datasets for personalization.
        self.personalized_trainset = None
        self.personalized_testset = None

        # Define the personalized model
        model_type = Config().algorithm.personalization.model_type
        model_params = Config().parameters.personalization.model._asdict()
        model_params["input_dim"] = self.model.encoding_dim
        model_params["output_dim"] = model_params["num_classes"]
        self.personalized_model = models_registry.get(
            model_name=Config().algorithm.personalization.model_name,
            model_type=model_type,
            model_params=model_params,
        )

    def set_personalized_datasets(self, trainset, testset):
        """Set the trainset."""
        self.personalized_trainset = trainset
        self.personalized_testset = testset

    def get_personalized_train_loader(
        self, batch_size, trainset=None, sampler=None, **kwargs
    ):
        """Obtain the trainset data loader for the personalization."""
        trainset = self.personalized_trainset

        return torch.utils.data.DataLoader(
            dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
        )

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """Obtain the training loader based on the learning mode."""
        if self.current_round > Config().trainer.rounds:
            return self.get_personalized_train_loader(
                batch_size, trainset, sampler, **kwargs
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

    def plato_ssl_loss_wrapper(self):
        """A wrapper to connect ssl loss with plato."""
        defined_ssl_loss = loss_criterion.get()

        def compute_plato_loss(outputs, labels):
            if isinstance(outputs, (list, tuple)):
                return defined_ssl_loss(*outputs)
            else:
                return defined_ssl_loss(outputs)

        return compute_plato_loss

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        if self.current_round > Config().trainer.rounds:
            return self.get_loss_criterion()

        return self.plato_ssl_loss_wrapper()

    def personalized_model_forward(self, examples, **kwargs):
        """Forward the input examples to the personalized model."""

        # Extract representation from the trained
        # frozen encoder of ssl.
        # No optimization is reuqired by this encoder.
        with torch.no_grad():
            features = self.model.encoder(examples)

        # Perfrom the training and compute the loss
        return self.personalized_model(features)

    def collect_data_encodings(self, data_loader):
        """Collecting the encodings of the data."""
        samples_encoding = None
        samples_label = None
        self.model.eval()
        self.model.to(self.device)
        for _, (examples, labels) in enumerate(data_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            with torch.no_grad():
                features = self.model.encoder(examples)
                if samples_encoding is None:
                    samples_encoding = features
                else:
                    samples_encoding = torch.cat([samples_encoding, features], dim=0)
                if samples_label is None:
                    samples_label = labels
                else:
                    samples_label = torch.cat([samples_label, labels], dim=0)

        return samples_encoding, samples_label

    def test_round_personalization(self, config, testset=None, sampler=None, **kwargs):
        """Test the personalization in each round.

        For SSL, the way to test the trained model before personalization is
        to use the KNN as a classifier to evaluate the extracted features.
        """
        logging.info("[Client #%d] Testing the model with KNN.", self.client_id)
        batch_size = config["batch_size"]

        train_loader = self.get_personalized_train_loader(batch_size=20)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler
        )

        train_encodings, train_labels = self.collect_data_encodings(train_loader)
        test_encodings, test_labels = self.collect_data_encodings(test_loader)

        # KNN.
        distances = torch.cdist(test_encodings, train_encodings, p=2)
        knn = distances.topk(1, largest=False)
        nearest_idx = knn.indices
        predicted_labels = train_labels[nearest_idx].view(-1)
        test_labels = test_labels.view(-1)

        # compute the accuracy
        num_correct = torch.sum(predicted_labels == test_labels).item()
        accuracy = num_correct / len(test_labels)

        return accuracy

    def test_model(self, config, testset, sampler=None, **kwargs):
        """Testing the model to report the accuracy of the
        personalized model."""
        if self.current_round > Config().trainer.rounds:
            return self.test_personalized_model(
                config, testset, sampler=sampler, **kwargs
            )
        else:
            return self.test_round_personalization(
                config, testset, sampler=sampler, **kwargs
            )
