"""
A base trainer to perform training and testing loops for self-supervised learning.
"""

import logging
from collections import UserList

import torch
from lightly.data.multi_view_collate import MultiViewCollate

from plato.config import Config
from plato.trainers import basic
from plato.models import registry as models_registry
from plato.trainers import optimizers, lr_schedulers, loss_criterion


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

        # Conversion to tensor to ensure backwards compatibility.
        labels = torch.tensor(labels, dtype=torch.long)

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
        model_params["input_dim"] = self.model.encoder.encoding_dim
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

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """Obtain the training loader based on the learning mode."""
        # Get the training loader for the personalization
        if self.current_round > Config().trainer.rounds:
            return torch.utils.data.DataLoader(
                dataset=self.personalized_trainset,
                shuffle=False,
                batch_size=batch_size,
                sampler=sampler,
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

    def get_optimizer(self, model):
        """Returns the optimizer for SSL and personalization."""
        if self.current_round <= Config().trainer.rounds:
            return super().get_optimizer(model)
        # Define the optimizer for the personalized model
        optimizer_name = Config().algorithm.personalization.optimizer
        optimizer_params = Config().parameters.personalization.optimizer._asdict()
        return optimizers.get(
            self.personalized_model,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
        )

    def get_ssl_criterion(self):
        """Get the loss criterion for SSL."""

        # Get loss criterion for the SSL
        ssl_loss_function = loss_criterion.get()

        def compute_plato_loss(outputs, labels):
            if isinstance(outputs, (list, tuple)):
                return ssl_loss_function(*outputs)
            else:
                return ssl_loss_function(outputs)

        return compute_plato_loss

    def get_loss_criterion(self):
        """Returns the loss criterion for the SSL."""
        # Get loss criterion for the personalization
        if self.current_round > Config().trainer.rounds:
            loss_criterion_type = Config().algorithm.personalization.loss_criterion
            loss_criterion_params = {}
            if hasattr(Config().parameters.personalization, "loss_criterion"):
                loss_criterion_params = (
                    Config().parameters.personalization.loss_criterion._asdict()
                )
            return loss_criterion.get(
                loss_criterion=loss_criterion_type,
                loss_criterion_params=loss_criterion_params,
            )

        return self.get_ssl_criterion()

    def get_lr_scheduler(self, config, optimizer):
        # Get the lr scheduler for the personalization
        if self.current_round > Config().trainer.rounds:
            lr_scheduler = Config().algorithm.personalization.lr_scheduler
            lr_params = Config().parameters.personalization.learning_rate._asdict()

            return lr_schedulers.get(
                optimizer,
                len(self.train_loader),
                lr_scheduler=lr_scheduler,
                lr_params=lr_params,
            )
        # Get the lr scheduler for the SSL
        return super().get_lr_scheduler(config, optimizer)

    def train_run_start(self, config):
        """Set the config before training."""
        if self.current_round > Config().trainer.rounds:
            config["batch_size"] = Config().algorithm.personalization.batch_size
            config["epochs"] = Config().algorithm.personalization.epochs

            self.personalized_model.to(self.device)
            self.personalized_model.train()

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop."""

        if not self.current_round > Config().trainer.rounds:
            return super().perform_forward_and_backward_passes(config, examples, labels)

        # Perform the local update on self.personalized_model
        self.optimizer.zero_grad()
        # Extract representation from the trained
        # frozen encoder of ssl.
        # No optimization is reuqired by this encoder.
        with torch.no_grad():
            features = self.model.encoder(examples)
        outputs = self.personalized_model(features)

        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss

    def collect_encodings(self, data_loader):
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

    def test_knn(self, config, testset=None, sampler=None, **kwargs):
        """Test the personalization in each round.

        For SSL, the way to test the trained model before personalization is
        to use the KNN as a classifier to evaluate the extracted features.
        """
        logging.info("[Client #%d] Testing the model with KNN.", self.client_id)
        batch_size = config["batch_size"]

        # Get the training loader and test loader
        train_loader = torch.utils.data.DataLoader(
            dataset=self.personalized_trainset,
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler
        )
        train_encodings, train_labels = self.collect_encodings(train_loader)
        test_encodings, test_labels = self.collect_encodings(test_loader)

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

    def test_personalized_model(self, config, testset=None, sampler=None, **kwargs):
        """Test the personalized model after the final round."""

        self.personalized_model.eval()
        self.personalized_model.to(self.device)

        self.model.eval()
        self.model.to(self.device)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=20, shuffle=False, sampler=sampler
        )

        correct = 0
        total = 0
        accuracy = 0
        with torch.no_grad():
            for _, (examples, labels) in enumerate(test_loader):
                examples, labels = examples.to(self.device), labels.to(self.device)

                features = self.model.encoder(examples)
                outputs = self.personalized_model(features)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total

        return accuracy

    def test_model(self, config, testset, sampler=None, **kwargs):
        """Testing the model to report the accuracy in each round."""
        if self.current_round > Config().trainer.rounds:
            return self.test_personalized_model(
                config, testset, sampler=sampler, **kwargs
            )
        else:
            return self.test_knn(config, testset, sampler=sampler, **kwargs)
