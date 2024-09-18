"""
An implementation of the FedMos algorithm.

X. Wang, Y. Chen, Y. Li, X. Liao, H. Jin and B. Li, "FedMoS: Taming Client Drift in Federated Learning with Double Momentum and Adaptive Selection," IEEE INFOCOM 2023

Paper: https://ieeexplore.ieee.org/document/10228957

Source code: https://github.com/Distributed-Learning-Networking-Group/FedMoS
"""
import copy

from plato.config import Config
from plato.trainers import basic

from optimizers import FedMosOptimizer

# pylint:disable=no-member
class Trainer(basic.Trainer):
    """
    FedMos's Trainer.
    """

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.local_param_tmpl = None

    def get_optimizer(self, model):
        """ Get the optimizer of the Fedmos."""
        a = Config().algorithm.a if hasattr(Config().algorithm, "a") else 0.9
        mu = Config().algorithm.mu if hasattr(Config().algorithm, "mu") else 0.9
        lr = Config().parameters.optimizer.lr if hasattr(Config().parameters.optimizer, "lr") else 0.01

        return FedMosOptimizer(model.parameters(), lr=lr, a=a, mu=mu)


    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop."""
        self.optimizer.zero_grad()

        outputs = self.model(examples)

        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.update_momentum()
        self.optimizer.step(copy.deepcopy(self.local_param_tmpl))

        return loss

    def train_run_start(self, config):
        super().train_run_start(config)
        # At the beginning of each round, the client records the local model
        self.local_param_tmpl = copy.deepcopy(self.model)
