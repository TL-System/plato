"""
A personalized federated learning trainer using Per-FedAvg
"""

import os
from typing import Iterator, Tuple, Union
from collections import OrderedDict
from copy import deepcopy

import torch

from plato.trainers import basic_personalized
from plato.utils.filename_formatter import NameFormatter


def get_data_batch(
    dataloader: torch.utils.data.DataLoader,
    iterator: Iterator,
    device=torch.device("cpu"),
):
    """Get one data batch from the dataloader."""
    try:
        samples, labels = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        samples, labels = next(iterator)

    return samples.to(device), labels.to(device)


def compute_gradients(
    model: torch.nn.Module,
    loss_criterion,
    data_batch: Tuple[torch.Tensor, torch.Tensor],
    base_grads: Union[Tuple[torch.Tensor, ...], None] = None,
    second_order_grads=False,
):
    """Compute the gradients."""
    examples, labels = data_batch
    if second_order_grads:
        frz_model_params = deepcopy(model.state_dict())
        delta = 1e-3
        dummy_model_params_1 = OrderedDict()
        dummy_model_params_2 = OrderedDict()
        with torch.no_grad():
            for (layer_name, param), grad in zip(model.named_parameters(), base_grads):
                dummy_model_params_1.update({layer_name: param + delta * grad})
                dummy_model_params_2.update({layer_name: param - delta * grad})

        model.load_state_dict(dummy_model_params_1, strict=False)
        logit_1 = model(examples)
        # loss_1 = loss_criterion(logit_1, y) / y.size(-1)
        loss_1 = loss_criterion(logit_1, labels)
        grads_1 = torch.autograd.grad(loss_1, model.parameters())

        model.load_state_dict(dummy_model_params_2, strict=False)
        logit_2 = model(examples)
        loss_2 = loss_criterion(logit_2, labels)
        # loss_2 = loss_criterion(logit_2, y) / y.size(-1)
        grads_2 = torch.autograd.grad(loss_2, model.parameters())

        model.load_state_dict(frz_model_params)

        grads = []
        with torch.no_grad():
            for g1, g2 in zip(grads_1, grads_2):
                grads.append((g1 - g2) / (2 * delta))
        return grads, loss_2

    else:
        logit = model(examples)
        # loss = loss_criterion(logit, y) / y.size(-1)
        loss = loss_criterion(logit, labels)
        grads = torch.autograd.grad(loss, model.parameters())
        return grads, loss


class Trainer(basic_personalized.Trainer):
    """A personalized federated learning trainer using the Per-FedAvg algorithm."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # the iterator for the dataloader
        self.iter_trainloader = None

    def train_epoch_start(self, config):
        """Method called at the beginning of a training epoch."""
        self.iter_trainloader = iter(self.train_loader)

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed.
        """
        alpha = config["alpha"]
        beta = config["beta"]
        if config["hessian_free"]:  # Per-FedAvg(HF)
            temp_model = deepcopy(self.model)

            grads, _ = compute_gradients(
                temp_model, self._loss_criterion, data_batch=(examples, labels)
            )
            for param, grad in zip(temp_model.parameters(), grads):
                param.data.sub_(alpha * grad)

            data_batch_2 = get_data_batch(
                self.train_loader, self.iter_trainloader, self.device
            )
            grads_1st, _ = compute_gradients(
                temp_model, self._loss_criterion, data_batch_2
            )

            data_batch_3 = get_data_batch(
                self.train_loader, self.iter_trainloader, self.device
            )

            grads_2nd, loss = compute_gradients(
                self.model,
                self._loss_criterion,
                data_batch_3,
                base_grads=grads_1st,
                second_order_grads=True,
            )

            for param, grad1, grad2 in zip(
                self.model.parameters(), grads_1st, grads_2nd
            ):
                param.data.sub_(beta * grad1 - beta * alpha * grad2)

        else:  # Per-FedAvg(FO)

            temp_model = deepcopy(self.model)

            grads, _ = compute_gradients(
                temp_model, self._loss_criterion, data_batch=(examples, labels)
            )
            for param, grad in zip(temp_model.parameters(), grads):
                param.data.sub_(alpha * grad)

            data_batch_2 = get_data_batch(
                self.train_loader, self.iter_trainloader, self.device
            )
            grads, loss = compute_gradients(
                temp_model, self._loss_criterion, data_batch_2
            )

            for param, grad in zip(self.model.parameters(), grads):
                param.data.sub_(beta * grad)

        self._loss_tracker.update(loss, labels.size(0))

        return loss

    def train_run_end(self, config):
        """Save the trained model to be the personalized model."""
        # copy the trained model to the personalized model
        self.personalized_model.load_state_dict(self.model.state_dict(), strict=True)

        current_round = self.current_round

        personalized_model_name = config["personalized_model_name"]
        save_location = self.get_checkpoint_dir_path()
        filename = NameFormatter.get_format_name(
            client_id=self.client_id,
            model_name=personalized_model_name,
            round_n=current_round,
            run_id=None,
            prefix="personalized",
            ext="pth",
        )
        os.makedirs(save_location, exist_ok=True)
        self.save_personalized_model(filename=filename, location=save_location)

    # pylint: disable=unused-argument
    def test_model(self, config, testset, sampler=None, **kwargs):
        """
        Evaluates the model with the provided test dataset and test sampler.

        Auguments:
        testset: the test dataset.
        sampler: the test sampler. The default is None.
        kwargs (optional): Additional keyword arguments.
        """
        accuracy = super().test_model(config, testset, sampler=None, **kwargs)

        # save the personaliation accuracy to the results dir
        self.checkpoint_personalized_accuracy(
            accuracy=accuracy,
            current_round=self.current_round,
            epoch=config["epochs"],
            run_id=None,
        )

        return accuracy

    def get_optimizer(self, model):
        """Returns the optimizer."""

    def get_lr_scheduler(self, config, optimizer):
        """Returns the learning rate scheduler, if needed."""

    def _adjust_lr(self, config, lr_scheduler, optimizer) -> torch.optim.Optimizer:
        """Returns an optimizer with an initial learning rate that has been
        adjusted according to the current round, so that learning rate
        schedulers can be effective throughout the communication rounds."""
        return None

    def personalized_train_model(
        self,
        config,
        trainset,
        sampler,
        **kwargs,
    ):
        """Perform personalized train process."""

        personalized_epochs = config["personalized_epochs"]
        batch_size = config["personalized_batch_size"]

        config["batch_size"] = batch_size
        config["epochs"] = personalized_epochs

        testset = kwargs["testset"]
        testset_sampler = kwargs["testset_sampler"]

        personalized_test_loader = self.get_personalized_data_loader(
            batch_size, testset, testset_sampler.get()
        )

        super().train_model(
            config,
            trainset,
            sampler,
            **kwargs,
        )
        eval_outputs = self.perform_evaluation(
            personalized_test_loader, self.personalized_model
        )
        accuracy = eval_outputs["accuracy"]

        # save the personaliation accuracy to the results dir
        self.checkpoint_personalized_accuracy(
            accuracy=accuracy,
            current_round=self.current_round,
            epoch=personalized_epochs,
            run_id=None,
        )
        if "max_concurrency" in config:

            # save the accuracy
            model_name = config["personalized_model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
            return None
