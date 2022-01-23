"""
A federated learning server using FedAsync.

Reference:

Xie, C., Koyejo, S., Gupta, I. (2019). "Asynchronous federated optimization,"
in Proc. 12th Annual Workshop on Optimization for Machine Learning (OPT 2020).

https://opt-ml.org/papers/2020/paper_28.pdf
"""
import logging
import os
from collections import OrderedDict

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAsync algorithm. """
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # The hyperparameter of FedAsync with a range of (0, 1)
        self.mixing_hyperparam = 1
        # Whether adjust mixing hyperparameter after each round
        self.adaptive_mixing = False

    def configure(self):
        """Configure the mixing hyperparameter for the server, as well as
        other parameters from config file.
        """
        super().configure()
        self.adaptive_mixing = hasattr(
            Config().server,
            'adaptive_mixing') and Config().server.adaptive_mixing

        if not hasattr(Config().server, 'mixing_hyperparameter'):
            logging.warning(
                "FedAsync: Variable mixing hyperparameter is required for the FedAsync server."
            )
        else:
            self.mixing_hyperparam = Config().server.mixing_hyperparameter
            if 0 < self.mixing_hyperparam < 1:
                logging.info("FedAsync: Mixing hyperparameter is set to %s.",
                             self.mixing_hyperparam)
            else:
                logging.warning(
                    "FedAsync: Invalid mixing hyperparameter. "
                    "The hyperparameter needs to be between 0 and 1 (exclusive)."
                )

    async def process_reports(self):
        """Process the client reports by aggregating their weights."""
        # Calculate the new mixing hyperparameter with client's staleness
        __, __, client_staleness = self.updates[0]

        if self.adaptive_mixing:
            self.mixing_hyperparam *= self.staleness_function(client_staleness)

        # Calculate updated weights from clients
        payload_received = [payload for (__, payload, __) in self.updates]
        weights_received = self.algorithm.compute_weight_updates(
            payload_received)

        # Actually update the global model's weights (PyTorch only)
        baseline_weights = self.algorithm.extract_weights()

        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight * (
                1 - self.mixing_hyperparam
            ) + weights_received[0][name] * self.mixing_hyperparam

        self.algorithm.load_weights(updated_weights)

        # Testing the global model accuracy
        if Config().clients.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                '[Server #{:d}] Average client accuracy: {:.2f}%.'.format(
                    os.getpid(), 100 * self.accuracy))
        else:
            # Testing the updated model directly at the server
            self.accuracy = await self.trainer.server_test(self.testset)
            logging.info(
                '[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
                    os.getpid(), 100 * self.accuracy))

        await self.wrap_up_processing_reports()

    @staticmethod
    def staleness_function(staleness) -> float:
        """ Polynomial staleness function as proposed in Sec. 5.2, Evaluation Setup. """
        a = 2
        return (staleness + 1)**-a
