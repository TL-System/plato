"""
A personalized federated learning trainer For APFL.
"""
import logging

import numpy as np

from pflbases import personalized_trainer


class Trainer(personalized_trainer.Trainer):
    """A trainer using the algorithm of APFL to jointly train the global
    and personalized models."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # the alpha used in the APFL paper
        self.alpha = 0.0
        self.adaptive_alpha = False

        # define the personalized optimizer
        # to update the personalized model
        self.personalized_optimizer = None

    def extract_alpha(self, loaded_status):
        """Extracting the alpha."""

        if (
            "learning" in loaded_status
            and loaded_status["learning"] is not None
            and "alpha" in loaded_status["learning"]
        ):
            self.alpha = loaded_status["learning"]["alpha"]
            logging.info(
                "[Client #%d] Loaded the alpha %s along with the personalized model",
                self.client_id,
                self.alpha,
            )
        else:
            logging.info(
                "[Client #%d] uses the initial alpha as no updated alpha exists.",
                self.client_id,
            )

    def update_alpha(self, eta):
        """Updating the alpha based on the Eq. 10 of the paper.

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
            self.model.parameters(), self.personalized_model.parameters()
        ):
            dif = p_params.data - l_params.data
            grad = (
                self.alpha * p_params.grad.data + (1 - self.alpha) * l_params.grad.data
            )
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))

        grad_alpha += 0.02 * self.alpha

        alpha_n = self.alpha - eta * grad_alpha
        self.alpha = np.clip(alpha_n.item(), 0.0, 1.0)

    def preprocess_personalized_model(self, config):
        """Do nothing to the loaded personalized model in APFL."""

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Performing forward and backward passes in the training loop.

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

    def train_run_start(self, config):
        """Defining items for personalization."""
        super().train_run_start(config)

        # define the personalized optimizer
        self.personalized_optimizer = self.get_personalized_optimizer()

        # set the personalized model to be trainable
        self.personalized_model.to(self.device)
        self.personalized_model.train()

        # initialize the alpha
        initial_alpha = config["alpha"]
        self.adaptive_alpha = config["adaptive_alpha"]
        self.alpha = initial_alpha if self.alpha == 0.0 else self.alpha

    def train_epoch_start(self, config):
        """Assigning the lr of optimizer to the personalized optimizer."""
        super().train_epoch_start(config)
        self.personalized_optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[
            0
        ]["lr"]

    def train_step_end(self, config, batch=None, loss=None):
        """Updating the alpha of APFL before each iteration."""
        super().train_step_end(config, batch, loss)
        # update alpha based on the Eq. 10 of the paper.
        if self.adaptive_alpha and self.current_epoch == 1 and batch == 0:
            # 0.1/np.sqrt(1+args.local_index))
            lr = self.lr_scheduler.get_lr()[0]
            previous_alpha = self.alpha
            self.update_alpha(lr)
            logging.info(
                "[Client #%d] in round#%d Update alpha from %.6f to %.6f.",
                self.client_id,
                self.current_round,
                previous_alpha,
                self.alpha,
            )
