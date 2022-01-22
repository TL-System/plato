"""
A federated learning server using FedAsync.

Reference:

Xie, C., Koyejo, S., & Gupta, I. (2019). "Asynchronous federated optimization." 
arXiv preprint arXiv:1903.03934.

https://opt-ml.org/papers/2020/paper_28.pdf
"""
import logging
import os

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAsync algorithm. """

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        # The hyperparameter of FedAsync with a range of (0,1)
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
                "FedAsync: Variable mixing hyperparameter is required for FedAsync server"
            )
        else:
            self.mixing_hyperparam = Config().server.mixing_hyperparameter
            if 0 < self.mixing_hyperparam < 1:
                logging.info("FedAsync: Mixing hyperparameter set to %s",
                             self.mixing_hyperparam)
            else:
                logging.warning(
                    "FedAsync: Invalid mixing hyperparameter. " +
                    "The hyperparameter needs to be within 0 and 1, exclusive")

    async def process_clients(self):
        """ Determine whether it is time to process the client reports and
            proceed with the aggregation process. """
        if len(self.reporting_clients) >= 1:
            logging.info("[Server #%d] %d client reports received.",
                         os.getpid(), len(self.reporting_clients))

            # Add client's report, payload, and staleness into updates
            client_info = self.reporting_clients[0]
            client_staleness = self.current_round - client_info[1][
                'starting_round']
            logging.info("[Server #%d] Processing received client report.",
                         os.getpid())
            self.updates.append((client_info[1]['report'],
                                 client_info[1]['payload'], client_staleness))

            await self.process_reports()
            await self.wrap_up()
            await self.select_clients()

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
        fedasync_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        for update in weights_received:
            for name, delta in update.items():
                # Use mixing parameter to update weights
                fedasync_update[name] += delta * self.mixing_hyperparam

        # Actually update the global model's weight
        updated_weights = self.algorithm.update_weights(fedasync_update)
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

    async def periodic_task(self):
        """ A periodic task that is executed from time to time"""
        # Because in FedAsync, the server aggregates weights whenever a client
        # reports back, so there is no need to check from time to time for
        # un-processed clients.
        return

    @staticmethod
    def staleness_function(staleness) -> float:
        a = 2
        return (staleness + 1) ** -a
