"""
A self-supervised learning (SSL) trainer for SSL training and testing.

Federated learning with SSL trains the global model based on the data loader and
objective function of SSL algorithms. For this unsupervised learning process, we
cannot test the model directly as the model only extracts features from the
data. Therefore, we use KNN as a classifier to get the accuracy of the global
model during the regular federated training process.

In the personalization process, each client trains a linear layer locally, based
on the features extracted by the trained global model.

The accuracy obtained by KNN during the regular federated training rounds may
not be used to compare with the accuracy in supervised learning methods. 
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
    """A container for SSL sample, which contains multiple views as a list."""

    def to(self, device):
        """Assign a list of views into the specific device."""
        for view_idx, view in enumerate(self.data):
            if isinstance(view, torch.Tensor):
                view = view.to(device)

            self[view_idx] = view

        return self.data


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


class Trainer(basic.Trainer):
    """A federated SSL trainer."""

    def __init__(self, model=None, callbacks=None):
        """Initialize the trainer."""
        super().__init__(model=model, callbacks=callbacks)

        # Datasets for personalization.
        self.personalized_trainset = None
        self.personalized_testset = None

        # Define the personalized model
        model_params = Config().parameters.personalization.model._asdict()
        model_params["input_dim"] = self.model.encoder.encoding_dim
        model_params["output_dim"] = model_params["num_classes"]
        self.local_layers = models_registry.get(
            model_name=Config().algorithm.personalization.model_name,
            model_type=Config().algorithm.personalization.model_type,
            model_params=model_params,
        )

    def set_personalized_datasets(self, trainset, testset):
        """Set the personalized trainset."""
        self.personalized_trainset = trainset
        self.personalized_testset = testset

    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """Obtain the training loader based on the learning mode."""
        # Get the trainloader for personalization
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
        """Return the optimizer for SSL and personalization."""
        if self.current_round <= Config().trainer.rounds:
            return super().get_optimizer(model)
        # Define the optimizer for the personalized model
        optimizer_name = Config().algorithm.personalization.optimizer
        optimizer_params = Config().parameters.personalization.optimizer._asdict()
        return optimizers.get(
            self.local_layers,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
        )

    def get_ssl_criterion(self):
        """
        Get the loss criterion for SSL. Some SSL algorithms, for example,
        BYOL, will overwrite this function for specific loss functions.
        """

        # Get loss criterion for the SSL
        ssl_loss_function = loss_criterion.get()

        # We need to wrap the loss function to make it compatible
        # with different types of outputs
        # The types of the outputs can vary from Tensor to a list of Tensors
        def compute_loss(outputs, __):
            if isinstance(outputs, (list, tuple)):
                return ssl_loss_function(*outputs)

            return ssl_loss_function(outputs)

        return compute_loss

    def get_loss_criterion(self):
        """Return the loss criterion for SSL."""
        # Get loss criterion for the subsequent training process
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
        # Get the lr scheduler for personalization
        if self.current_round > Config().trainer.rounds:
            lr_scheduler = Config().algorithm.personalization.lr_scheduler
            lr_params = Config().parameters.personalization.learning_rate._asdict()

            return lr_schedulers.get(
                optimizer,
                len(self.train_loader),
                lr_scheduler=lr_scheduler,
                lr_params=lr_params,
            )
        # Get the lr scheduler for SSL
        return super().get_lr_scheduler(config, optimizer)

    def train_run_start(self, config):
        """Set the config before training."""
        if self.current_round > Config().trainer.rounds:
            # Set the config for the personalization
            config["batch_size"] = Config().algorithm.personalization.batch_size
            config["epochs"] = Config().algorithm.personalization.epochs

            # Move the local layers to the device and set it to train mode
            self.local_layers.to(self.device)
            self.local_layers.train()

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.
        This function needs to reuse the optimization code of Plato as
        during personalization, the encoder of the self.model will be used to
        extract features into the local layers.
        """

        # Perform SSL training in the first `Config().trainer.rounds`` rounds
        if not self.current_round > Config().trainer.rounds:
            return super().perform_forward_and_backward_passes(config, examples, labels)

        # Perform personalization after the final round
        # Perform the local update on self.local_layers
        self.optimizer.zero_grad()

        # Use the trained encoder to output features.
        # No optimizer for this basic encoder
        features = self.model.encoder(examples)
        outputs = self.local_layers(features)

        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss

    def collect_encodings(self, data_loader):
        """Collect encodings of the data by using self.model."""
        samples_encoding = None
        samples_label = None
        self.model.eval()
        self.model.to(self.device)
        for examples, labels in data_loader:
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

    def test_model(self, config, testset, sampler=None, **kwargs):
        """Test the model to report the accuracy in each round."""
        batch_size = config["batch_size"]
        if self.current_round > Config().trainer.rounds:
            # Test the personalized model after the final round.
            self.local_layers.eval()
            self.local_layers.to(self.device)

            self.model.eval()
            self.model.to(self.device)

            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, sampler=sampler
            )

            correct = 0
            total = 0
            accuracy = 0
            with torch.no_grad():
                for examples, labels in test_loader:
                    examples, labels = examples.to(self.device), labels.to(self.device)

                    features = self.model.encoder(examples)
                    outputs = self.local_layers(features)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total

            return accuracy
        else:
            # Test the personalized model in each round.

            # For SSL, the way to test the trained model before personalization is
            # to use the KNN as a classifier to evaluate the extracted features.

            logging.info("[Client #%d] Testing the model with KNN.", self.client_id)

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
            # For evaluating self-supervised performance, we need to calculate
            # distance between training samples and testing samples.
            train_encodings, train_labels = self.collect_encodings(train_loader)
            test_encodings, test_labels = self.collect_encodings(test_loader)

            # Build KNN and perform the prediction
            distances = torch.cdist(test_encodings, train_encodings, p=2)
            knn = distances.topk(1, largest=False)
            nearest_idx = knn.indices
            predicted_labels = train_labels[nearest_idx].view(-1)
            test_labels = test_labels.view(-1)

            # Compute the accuracy
            num_correct = torch.sum(predicted_labels == test_labels).item()
            accuracy = num_correct / len(test_labels)

            return accuracy
