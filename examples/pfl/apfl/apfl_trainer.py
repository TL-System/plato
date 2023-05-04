"""
A personalized federated learning trainer using APFL.

"""

import os
import logging

import numpy as np

from plato.trainers import basic_personalized
from plato.utils.filename_formatter import NameFormatter


class Trainer(basic_personalized.Trainer):
    """A personalized federated learning trainer using the APFL algorithm."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # the alpha used in the APFL paper
        self.alpha = 0.0
        self.is_adaptive_alpha = False

    def extract_alpha(self, loaded_status):
        """Extract the alpha."""

        if (
            "learning" in loaded_status
            and loaded_status["learning"] is not None
            and "alpha" in loaded_status["learning"]
        ):
            self.alpha = loaded_status["learning"]["alpha"]
            logging.info(
                "[Client #%s] Loaded the alpha %s along with the personalized model",
                self.client_id,
                self.alpha,
            )
        else:
            logging.info(
                "[Client #%s] uses the initial alpha as no updated alpha exists."
            )

    def update_alpha(self, defined_model, personalized_model, alpha, eta):
        """Update the alpha based on the Eq. 10 of the paper.

        The implementation of this alpha update comes from the
        APFL code of:
         https://github.com/MLOPTPSU/FedTorch/blob/main/main.py

        The only concern is that
            why 'grad_alpha' needs to be computed as:
                grad_alpha = grad_alpha + 0.02 * alpha
        """
        grad_alpha = 0
        # perform the second term of Eq. 10
        for l_params, p_params in zip(
            defined_model.parameters(), personalized_model.parameters()
        ):
            dif = p_params.data - l_params.data
            grad = alpha * p_params.grad.data + (1 - alpha) * l_params.grad.data
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))

        grad_alpha += 0.02 * alpha

        alpha_n = alpha - eta * grad_alpha
        alpha_n = np.clip(alpha_n.item(), 0.0, 1.0)

        return alpha_n

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

        # initialize the alpha
        initial_alpha = config["alpha"]
        self.is_adaptive_alpha = config["is_adaptive_alpha"]
        self.alpha = initial_alpha if self.alpha == 0.0 else self.alpha

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed.

        This implementation refers to:
        https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py

        It seems that in this implementation, the v^{t} will be optimized based on the
        optimized w^{t}, instead of the w^{t-1} shown in Table Algorithm 1 of the paper.

        To fix the issue, we have the `perform_forward_and_backward_passes_V2`.

        However, to be consistent with FedTorch's implementation, this version is
        used by default.
        """

        # perform normal local update
        super().perform_forward_and_backward_passes(config, examples, labels)

        # perform personalization
        # clean the grads in normal optimizer
        self.optimizer.zero_grad()
        self.personalized_optimizer.zero_grad()

        output1 = self.personalized_model(examples)
        output2 = self.model(examples)
        output = self.alpha * output1 + (1 - self.alpha) * output2
        personalized_loss = self._loss_criterion(output, labels)

        personalized_loss.backward()
        self.personalized_optimizer.step()

        return personalized_loss

    def perform_forward_and_backward_passes_v2(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed.

        This implementation refers to:
        https://github.com/lgcollins/FedRep

        """

        # perform personalization
        # clean the grads in normal optimizer
        self.optimizer.zero_grad()
        self.personalized_optimizer.zero_grad()

        output1 = self.personalized_model(examples)
        output2 = self.model(examples)
        output = self.alpha * output1 + (1 - self.alpha) * output2
        personalized_loss = self._loss_criterion(output, labels)

        personalized_loss.backward()
        self.personalized_optimizer.step()

        # perform normal local update
        super().perform_forward_and_backward_passes(config, examples, labels)

        return personalized_loss

    def train_step_end(self, config, batch=None, loss=None):
        """Update the alpha when possible."""

        # update alpha based on the Eq. 10 of the paper.
        if self.is_adaptive_alpha and self.current_epoch == 1 and batch == 0:
            # 0.1/np.sqrt(1+args.local_index))
            lr = self.lr_scheduler.get_lr()[0]
            previous_alpha = self.alpha
            self.alpha = self.update_alpha(
                self.model, self.personalized_model, self.alpha, lr
            )
            logging.info(
                "[Client #%d] in round#%d Update alpha from %.6f to %.6f.",
                self.client_id,
                self.current_round,
                previous_alpha,
                self.alpha,
            )

    def train_run_end(self, config):
        """Saving the personalized model and alpha."""
        # save the personalized model for current round
        # to the model dir of this client
        if "max_concurrency" in config:

            current_round = self.current_round

            learning_dict = {"alpha": self.alpha}
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

    def personalized_train_model(self, config, trainset, sampler, **kwargs):
        """APFL will only evaluate the personalized model."""
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
