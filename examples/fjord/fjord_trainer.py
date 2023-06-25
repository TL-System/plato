"""
FjORD algorithm trainer.
"""
import random
import torch
import numpy as np
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
        self.rates = np.array([0.0625, 0.125, 0.25, 0.5, 1.0])

    def perform_forward_and_backward_passes(self, config, examples, labels):
        self.optimizer.zero_grad()

        outputs = self.model(examples)
        kdloss_func = torch.nn.KLDivLoss()
        sample_rate = random.choice(self.rates[self.rates.tolist().find(self.rate) :])
        subnet = self.model_class(
            model_rare=sample_rate, **Config().parameters.client_model._asdict()
        ).to(Config.device())
        subnet_outputs = subnet(examples).detach()

        loss = self._loss_criterion(outputs, labels) + kdloss_func(
            outputs, subnet_outputs
        )
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss
