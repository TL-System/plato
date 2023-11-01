"""
A personalized federated learning trainer using Ditto.
"""
import os
import logging

import torch

from plato.trainers import tracking, basic
from plato.utils import fonts
from plato.config import Config
from plato.models import registry as models_registry


class Trainer(basic.Trainer):
    """A personalized federated learning trainer using the Ditto algorithm."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)

        # the lambda used in the Ditto paper
        self.ditto_lambda = Config().algorithm.ditto_lambda
        # The personalized model is the vnet defined in the paper
        if model is None:
            self.personalized_model = models_registry.get()
        else:
            self.personalized_model = model()
        # The global model weights received from the server.
        #   which is the w^t in the paper.
        self.initial_wnet_params = None

    def train_run_start(self, config):
        super().train_run_start(config)
        self.initial_wnet_params = self.model.cpu().state_dict()

    def train_run_end(self, config):
        """Performing the personalized learning of Ditto."""
        super().train_run_end(config)

        logging.info(
            fonts.colourize(
                "[Client #%d] performing Ditto Solver for personalizaiton: ",
                colour="blue",
            ),
            self.client_id,
        )

        # load v net parameters from filesystem
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name
        filename = f"{model_path}/{model_name}_{self.client_id}_v_net.pth"
        if os.path.exists(filename):
            self.personalized_model.load_state_dict(
                torch.load(filename, map_location=torch.device("cpu"))
            )

        personalized_optimizer = self.get_optimizer(self.personalized_model)
        lr_scheduler = self.get_lr_scheduler(config, personalized_optimizer)
        epoch_loss_meter = tracking.LossTracker()

        self.personalized_model.to(self.device)
        self.personalized_model.train()
        for epoch in range(1, config["epochs"] + 1):
            epoch_loss_meter.reset()
            for _, (examples, labels) in enumerate(self.train_loader):
                examples, labels = examples.to(self.device), labels.to(self.device)

                # Clear the previous gradient
                personalized_optimizer.zero_grad()

                ## 1.- Compute the ∇F_k(v_k), thus to compute the first term
                #   of the equation in the Algorithm. 1.
                # i.e., v_k − η∇F_k(v_k)
                # This can be achieved by the general optimization step.
                # Perfrom the training and compute the loss
                preds = self.personalized_model(examples)
                loss = self._loss_criterion(preds, labels)

                # Perfrom the optimization
                loss.backward()
                personalized_optimizer.step()
                ## 2.- Compute the ηλ(v_k − w^t), which is the second term of
                #   the corresponding equation in Algorithm. 1.
                lr = personalized_optimizer.param_groups[0]["lr"]

                for (
                    v_net_name,
                    v_net_param,
                ) in self.personalized_model.named_parameters():
                    v_net_param.data = v_net_param.data - lr * self.ditto_lambda * (
                        v_net_param.data
                        - self.initial_wnet_params[v_net_name].to(self.device)
                    )

                # Update the epoch loss container
                epoch_loss_meter.update(loss, labels.size(0))

            lr_scheduler.step()
            logging.info(
                "[Client #%d] Personalization Training Epoch: [%d/%d]\tLoss: %.6f",
                self.client_id,
                epoch,
                config["epochs"],
                epoch_loss_meter.average,
            )
        self.personalized_model.to(torch.device("cpu"))
        # save v net parameters from filesystem
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name
        filename = f"{model_path}/{model_name}_{self.client_id}_v_net.pth"
        torch.save(self.personalized_model.state_dict(), filename)

        # In the final personalization round, do the fine-tune and testing
        #   on the local personalized model
        if self.current_round > Config().trainer.rounds:
            self.model = self.personalized_model
