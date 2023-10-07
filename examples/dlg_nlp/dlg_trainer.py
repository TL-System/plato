import math
import pickle
import random

import numpy as np
import torch
from plato.config import Config
from plato.trainers import basic
from torchvision import transforms

from utils.utils import cross_entropy_for_onehot, label_to_onehot

criterion = cross_entropy_for_onehot
tt = transforms.ToPILImage()


class Trainer(basic.Trainer):
    """The federated learning trainer for the gradient leakage attack."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        callbacks: The callbacks that this trainer uses.
        """

        def weights_init(m):
            """Initializing the weights and biases in the model."""
            if hasattr(m, "weight"):
                m.weight.data.uniform_(-0.5, 0.5)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data.uniform_(-0.5, 0.5)

        super().__init__(model=model, callbacks=callbacks)

        # DLG explicit weights initialziation
        if (
            hasattr(Config().algorithm, "init_params")
            and Config().algorithm.init_params
        ):
            self.model.apply(weights_init)

        self.examples = None
        self.trainset = None
        self.full_examples = None
        self.full_labels = None
        self.full_onehot_labels = None
        self.list_grad = None
        self.target_grad = None
        self.feature_fc1_graph = None
        self.sensitivity = None

    def train_run_start(self, config):
        """Method called at the start of training run."""
        self.target_grad = None

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop."""
        # Store data in the first epoch (later epochs will still have the same partitioned data)
        if self.current_epoch == 1:
            try:
                self.full_examples = torch.cat((examples, self.full_examples), dim=0)
                self.full_labels = torch.cat((labels, self.full_labels), dim=0)
            except:
                self.full_examples = examples
                self.full_labels = labels

            self.full_onehot_labels = label_to_onehot(
                self.full_labels, num_classes=Config().parameters.model.num_classes
            )

        examples.requires_grad = True
        self.examples = examples
        self.model.zero_grad()

        # Compute gradients in the current step
        outputs, self.feature_fc1_graph = self.model(examples)

        # Save the ground truth and gradients
        loss = self._loss_criterion(outputs, labels)
        grad = torch.autograd.grad(
            loss,
            self.model.parameters(),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )
        self.list_grad = list((_.detach().clone() for _ in grad))

        self._loss_tracker.update(loss, labels.size(0))

        return loss

    def train_step_end(self, config, batch=None, loss=None):
        """Method called at the end of a training step."""
        # cast grad back to tuple type
        grad = tuple(self.list_grad)
        
        # Update model weights with gradients and learning rate
        for param, grad_part in zip(self.model.parameters(), grad):
            param.data = param.data - Config().parameters.optimizer.lr * grad_part

        # Sum up the gradients for each local update
        try:
            self.target_grad = [
                sum(x)
                for x in zip(list((_.detach().clone() for _ in grad)), self.target_grad)
            ]
        except:
            self.target_grad = list((_.detach().clone() for _ in grad))

    def train_run_end(self, config, **kwargs):
        """Method called at the end of a training run."""
        if (
            hasattr(Config().algorithm, "share_gradients")
            and Config().algorithm.share_gradients
        ):
            try:
                total_local_steps = config["epochs"] * math.ceil(
                    Config().data.partition_size / config["batch_size"]
                )
                self.target_grad = [x / total_local_steps for x in self.target_grad]
            except:
                self.target_grad = None

        self.full_examples = self.full_examples.detach()
        file_path = f"{Config().params['model_path']}/{self.client_id}.pickle"
        with open(file_path, "wb") as handle:
            pickle.dump(
                [self.full_examples, self.full_onehot_labels, self.target_grad], handle
            )

    @staticmethod
    def process_outputs(outputs):
        """
        Method called after the model updates have been generated.
        """
        return outputs[0]
