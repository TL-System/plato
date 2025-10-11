"""
FjORD algorithm trainer.
"""

import copy
import random

import numpy as np
import torch

from plato.config import Config
from plato.trainers.basic import Trainer


class ServerTrainer(Trainer):
    """A federated learning trainer of FjORD, used by the server."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        self.model = model(**Config().parameters.model._asdict())


class ClientTrainer(Trainer):
    """A federated learning trainer of FjORD, used by the client."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.rate = 1.0
        self.rates = np.array([1, 0.5, 0.25, 0.125, 0.0625])
        self.model_class = model

    def perform_forward_and_backward_passes(self, config, examples, labels):
        self.optimizer.zero_grad()

        outputs = self.model(examples)
        kdloss_func = torch.nn.MSELoss()
        sample_rate = random.choice(self.rates[self.rates.tolist().index(self.rate) :])
        subnet = self.model_class(
            model_rate=sample_rate, **Config().parameters.client_model._asdict()
        )
        local_parameters = subnet.state_dict()
        for key, value in self.model.state_dict().items():
            if "weight" in key or "bias" in key:
                if value.dim() == 4 or value.dim() == 2:
                    local_parameters[key] = copy.deepcopy(
                        value[
                            : local_parameters[key].shape[0],
                            : local_parameters[key].shape[1],
                            ...,
                        ]
                    )
                else:
                    local_parameters[key] = copy.deepcopy(
                        value[: local_parameters[key].shape[0]]
                    )
        subnet.load_state_dict(local_parameters)
        subnet = subnet.to(Config.device())
        subnet_outputs = subnet(examples)  # .detach()

        loss = self._loss_criterion(outputs, labels)
        kdloss = kdloss_func(outputs, subnet_outputs)
        loss = loss + kdloss
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss
