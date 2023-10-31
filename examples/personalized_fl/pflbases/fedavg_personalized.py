"""
A base server to perform personalized federated learning.

This server is able to perform personalization in all clients after the 
final round.
"""

from typing import List
import random
import logging

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """Federated learning server for personalization and partial client selection."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # The number of clients that will be selected
        # in federated learning
        self.number_participating_clients = 0

        # The ids of these selected clients
        self.participating_clients_pool = None

        # Whether personalization has been started
        self.personalization_started = False

        self.initialize_personalization()

    def initialize_personalization(self):
        """Initialize two types of clients."""

        ## 1. Initialize participanting client ratio
        participating_client_ratio = (
            Config().algorithm.personalization.participating_client_ratio
            if hasattr(Config().algorithm.personalization, "participating_client_ratio")
            else 1.0
        )

        #  clients will / will not participant in federated training
        self.number_participating_clients = int(
            self.total_clients * participating_client_ratio
        )

        logging.info(
            "[%s] Total clients (%d), participanting clients (%d), "
            "participanting ratio (%.3f).",
            self,
            self.total_clients,
            self.number_participating_clients,
            participating_client_ratio,
        )

    def set_clients_pool(self, clients_pool: List[int]):
        """Set clients pool utilized in federated learning.

        Note: the participating clients pool will be set in the first round and no
            modification is performed afterwards.
        """

        # Only need to set the clients pool when they are empty.
        if self.participating_clients_pool is None:
            self.participating_clients_pool = clients_pool[
                : self.number_participating_clients
            ]

            logging.info(
                "[%s] Prepared participating clients pool: %s",
                self,
                self.participating_clients_pool,
            )

    def get_normal_clients(self, clients_pool: List[int], clients_count: int):
        """Operations to guarantee general federated learning without personalization."""

        # Reset `clients_per_round` to the predefined hyper-parameter
        self.clients_per_round = Config().clients.per_round

        # Set the clients_pool to be participating_clients_pool
        clients_pool = self.participating_clients_pool
        clients_count = self.clients_per_round

        # By default, when we run the general federated training,
        # the clients pool should be participating clients
        assert clients_count <= len(self.participating_clients_pool)

        return clients_pool, clients_count

    def get_personalization_clients(self):
        """Performing personalization after the final round."""

        logging.info("Starting personalization after the final round.")

        # To terminate the personalization afterwards
        self.personalization_started = True

        # Do personalization on all clients
        return self.clients_pool, len(self.clients_pool)

    def get_clients(self, clients_pool: List[int], clients_count: int):
        """Determine clients pool and clients count before samling clients."""

        # Perform normal training
        clients_pool, clients_count = self.get_normal_clients(
            clients_pool, clients_count
        )
        # Perform personalization
        if self.current_round > Config().trainer.rounds:
            clients_pool, clients_count = self.get_personalization_clients()

        return clients_pool, clients_count

    def choose_clients(self, clients_pool: List[int], clients_count: int):
        """Chooses a subset of the clients to participate in each round.

        In plato, this input `clients_pool` contains total clients
        id by default.
        """
        # Set required clients pool when possible
        self.set_clients_pool(clients_pool)

        clients_pool, clients_count = self.get_clients(clients_pool, clients_count)

        random.setstate(self.prng_state)

        # Select clients randomly
        selected_clients = random.sample(clients_pool, clients_count)

        self.prng_state = random.getstate()

        logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients

    async def wrap_up(self):
        """Wrapping up when each round of training is done."""
        self.save_to_checkpoint()

        if self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")

            if self.personalization_started:
                logging.info(
                    "Personalization completed after the final round.",
                )
                await self._close()
