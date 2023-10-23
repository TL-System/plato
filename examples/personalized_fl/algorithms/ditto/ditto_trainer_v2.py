"""
A personalized federated learning trainer using Ditto.

This implementation striclty follows the official code presented in
https://github.com/lgcollins/FedRep.

"""

import copy
import logging


from plato.trainers import tracking
from plato.utils import fonts
from plato.config import Config


from pflbases import personalized_trainer


class Trainer(personalized_trainer.Trainer):
    """A personalized federated learning trainer using the Ditto algorithm."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # the lambda used in the Ditto paper
        self.ditto_lambda = Config().algorithm.ditto_lambda
        self.initial_model_params = None

        self.personalized_optimizer = None

    def preprocess_models(self, config):
        """Do nothing to the loaded personalized model in Ditto."""

    def train_run_start(self, config):
        """Defining the personalization before running."""
        super().train_run_start(config)

        if self.do_final_personalization:
            config["epochs"] = 0

        # initialize the optimizer, lr_schedule, and loss criterion
        self.personalized_optimizer = self.get_personalized_optimizer()

        self.personalized_model.to(self.device)
        self.personalized_model.train()

        # backup the unoptimized global model
        # this is used as the baseline ditto weights in the Ditto solver
        self.initial_model_params = copy.deepcopy(self.model.state_dict())

    def train_epoch_start(self, config):
        """Assigning the lr of optimizer to the personalized optimizer."""
        super().train_epoch_start(config)
        self.personalized_optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[
            0
        ]["lr"]

    def train_run_end(self, config):
        """Performing the personalized learning of Ditto."""
        # save the personalized model for current round
        # to the model dir of this client
        personalized_epochs = config["epochs"]
        if hasattr(Config().algorithm.personalization, "epochs"):
            personalized_epochs = Config().algorithm.personalization.epochs

        # do nothing in the final personalization
        if self.do_final_personalization:
            return

        logging.info(
            fonts.colourize(
                "[Client #%d] performing Ditto Solver for personalizaiton: ",
                colour="blue",
            ),
            self.client_id,
        )
        epoch_loss_meter = tracking.LossTracker()

        for epoch in range(1, personalized_epochs + 1):
            epoch_loss_meter.reset()
            for _, (examples, labels) in enumerate(self.train_loader):
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
                    self.initial_model_params[key] = self.initial_model_params[key].to(
                        self.device
                    )

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

        super().train_run_end(config)

    def postprocess_models(self, config):
        """Do nothing to the personalized model in Ditto."""
