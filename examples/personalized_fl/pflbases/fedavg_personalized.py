"""
A base server to perform personalized federated learning.
"""

from typing import List
import random
import logging

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """A base server to control how many clients will participate in the learning and
    enable a final personalization round."""

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

        # The number of clients that will participate
        # in federated learning
        self.number_participating_clients = 0

        # The IDs of these participating clients
        self.participating_clients_pool = None

        # Whether personalization has been started
        self.personalization_started = False

        self.initialize_participant()

    def initialize_participant(self):
        """Initialize participation information."""

        ## Initialize participanting client ratio
        participating_client_ratio = (
            Config().algorithm.personalization.participating_client_ratio
            if hasattr(Config().algorithm.personalization, "participating_client_ratio")
            else 1.0
        )

        #  Compute how many clients will participate in federated learning
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
        """Set clients pool utilized in federated learning."""

        # The participating client pool will be set in the first round,
        # and no modification will be performed afterward.
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
        """Get the clients used in normal federated training rounds."""

        # Use the participating clients pool
        clients_pool = self.participating_clients_pool
        clients_count = self.clients_per_round

        return clients_pool, clients_count

    def get_personalization_clients(self):
        """Get clients used in the final personalization."""
        # Use all clients in the final personalization
        self.clients_per_round = self.total_clients

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
            self.personalization_started = True
            clients_pool, clients_count = self.get_personalization_clients()

        return clients_pool, clients_count

    def choose_clients(self, clients_pool: List[int], clients_count: int):
        """Choose a subset of the clients to participate in each round."""
        # Set required client pools when possible
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

        # Continue training until the final personalization round
        # has been completed
        if self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")

            if self.personalization_started:
                logging.info(
                    "Personalization completed after the final round.",
                )
                await self._close()
