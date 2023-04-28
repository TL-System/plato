"""
A personalized federated learning trainer using Ditto.

This implementation striclty follows the official code presented in
https://github.com/lgcollins/FedRep.

"""

import os
import copy
import logging

from tqdm import tqdm

from plato.trainers import basic_personalized
from plato.utils.filename_formatter import NameFormatter
from plato.trainers import tracking
from plato.utils import fonts


class Trainer(basic_personalized.Trainer):
    """A personalized federated learning trainer using the Ditto algorithm."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # the lambda used in the Ditto paper
        self.ditto_lambda = 0.0
        self.initial_model_params = None

    def train_run_start(self, config):
        """Define personalization before running."""

        # initialize the optimizer, lr_schedule, and loss criterion
        self.personalized_optimizer = self.get_personalized_optimizer(
            self.personalized_model
        )
        self.personalized_optimizer = self._adjust_lr(
            config, self.lr_scheduler, self.personalized_optimizer
        )

        self.personalized_model.to(self.device)
        self.personalized_model.train()

        # initialize the lambda
        self.ditto_lambda = config["ditto_lambda"]
        # backup the unoptimized global model
        # this is used as the baseline ditto weights in the Ditto solver
        self.initial_model_params = copy.deepcopy(self.model.state_dict())

    def train_run_end(self, config):
        """Perform the personalized learning of Ditto."""
        # save the personalized model for current round
        # to the model dir of this client
        personalized_epochs = config["personalized_epochs"]

        show_str = logging.info(
            fonts.colourize("[Client #%d] performing Ditto Solver: ", colour="blue"),
            self.client_id,
        )
        global_progress = tqdm(range(1, personalized_epochs + 1), desc=show_str)
        epoch_loss_meter = tracking.LossTracker()

        for epoch in global_progress:
            epoch_loss_meter.reset()
            local_progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{personalized_epochs+1}",
                disable=True,
            )
            for _, (examples, labels) in enumerate(local_progress):
                examples, labels = examples.to(self.device), labels.to(self.device)
                # backup the params of defined model before optimization
                # this is the v_k in the Algorithm. 1
                v_initial = copy.deepcopy(self.personalized_model.state_dict())

                # Clear the previous gradient
                self.personalized_optimizer.zero_grad()

                ## 1.- Compute the ∇F_k(v_k), thus to compute the first term
                #   of the equation in the Algorithm. 1.
                # i.e., v_k − η∇F_k(v_k)
                # This can be achieved by the general optimization step.
                # Perfrom the training and compute the loss
                # Perfrom the training and compute the loss
                preds = self.personalized_model(examples)
                loss = self._loss_criterion(preds, labels)

                # Perfrom the optimization
                loss.backward()
                self.personalized_optimizer.step()
                ## 2.- Compute the ηλ(v_k − w^t), which is the second term of
                #   the corresponding equation in Algorithm. 1.
                w_net = copy.deepcopy(self.personalized_model.state_dict())
                lr = self.lr_scheduler.get_lr()[0]
                for key in w_net.keys():
                    w_net[key] = w_net[key] - lr * self.ditto_lambda * (
                        v_initial[key] - self.initial_model_params[key]
                    )

                self.personalized_model.load_state_dict(w_net)
                # Update the epoch loss container
                epoch_loss_meter.update(loss, labels.size(0))

            logging.info(
                "[Client #%d] Personalization Training Epoch: [%d/%d]\tLoss: %.6f",
                self.client_id,
                epoch,
                personalized_epochs,
                epoch_loss_meter.average,
            )

        if "max_concurrency" in config:

            current_round = self.current_round

            learning_dict = {"lambda": self.ditto_lambda}
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
            self.save_personalized_model(
                filename=filename, location=save_location, learning_dict=learning_dict
            )

    # pylint: disable=unused-argument
    def test_model(self, config, testset, sampler=None, **kwargs):
        """
        Evaluates the model with the provided test dataset and test sampler.
        Auguments:
        testset: the test dataset.
        sampler: the test sampler. The default is None.
        kwargs (optional): Additional keyword arguments.
        """
        trained_model_params = copy.deepcopy(self.model)
        self.model.load_state_dict(self.personalized_model.state_dict(), strict=True)
        accuracy = super().test_model(config, testset, sampler=None, **kwargs)
        self.model.load_state_dict(trained_model_params, strict=True)
        # save the personaliation accuracy to the results dir
        self.checkpoint_personalized_accuracy(
            accuracy=accuracy,
            current_round=self.current_round,
            epoch=config["epochs"],
            run_id=None,
        )

        return accuracy

    def personalized_train_model(self, config, trainset, sampler, **kwargs):
        """Ditto will only evaluate the personalized model."""
        batch_size = config["batch_size"]

        testset = kwargs["testset"]
        testset_sampler = kwargs["testset_sampler"]

        personalized_test_loader = self.get_personalized_data_loader(
            batch_size, testset, testset_sampler.get()
        )

        eval_outputs = self.perform_evaluation(
            personalized_test_loader, self.personalized_model
        )
        accuracy = eval_outputs["accuracy"]

        # save the personaliation accuracy to the results dir
        self.checkpoint_personalized_accuracy(
            accuracy=accuracy, current_round=self.current_round, epoch=0, run_id=None
        )

        if "max_concurrency" in config:

            # save the accuracy directly for latter usage
            # in the eval_test(...)
            model_name = config["personalized_model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
            return None
