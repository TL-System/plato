"""
A personalized federated learning trainer with Ditto.
"""
import os
import copy
import logging

import torch

from plato.trainers import tracking, basic
from plato.utils import fonts
from plato.config import Config
from plato.models import registry as models_registry


class Trainer(basic.Trainer):
    """
    A trainer with Ditto, which first trains the global model and then trains
    the personalized model.
    """

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        # The lambda adjusts the gradients
        self.ditto_lambda = Config().algorithm.ditto_lambda

        # Get the personalized model
        if model is None:
            self.personalized_model = models_registry.get()
        else:
            self.personalized_model = model()

        # The global model weights, which is w^t in the paper
        self.initial_wnet_params = None

    def train_run_start(self, config):
        super().train_run_start(config)

        # Make a copy of the model before local training starts, which will be used when optimizing
        # the personalized model
        self.initial_wnet_params = copy.deepcopy(self.model.cpu().state_dict())

    def train_run_end(self, config):
        """
        Optimize the personalized model for epochs following Algorithm 1.
        """
        super().train_run_end(config)

        logging.info(
            fonts.colourize(
                "[Client #%d] Started Ditto's personalized training.",
                colour="blue",
            ),
            self.client_id,
        )

        # Load personalized model
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
            for __, (examples, labels) in enumerate(self.train_loader):
                examples, labels = examples.to(self.device), labels.to(self.device)
                personalized_optimizer.zero_grad()

                # Compute the first term of the equation in Algorithm 1: v_k − η∇F_k(v_k)
                preds = self.personalized_model(examples)
                loss = self._loss_criterion(preds, labels)

                loss.backward()
                personalized_optimizer.step()

                # Compute ηλ(v_k − w^t), which is the second term in Algorithm 1
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

        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name
        filename = f"{model_path}/{model_name}_{self.client_id}_v_net.pth"
        torch.save(self.personalized_model.state_dict(), filename)

        # In the final personalization round, test the personalized model only
        if self.current_round > Config().trainer.rounds:
            self.model = self.personalized_model
