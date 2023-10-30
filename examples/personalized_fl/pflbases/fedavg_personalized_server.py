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
from plato.utils import fonts


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

        # The number of participanted clients
        self.number_participating_clients = 0

        # The list of participanted clients
        self.participating_clients_pool = None

        # Whether stop the terminate personalization afterwards
        self.to_terminate_personalization = False

        self.initialize_personalization()

    def initialize_personalization(self):
        """Initialize two types of clients."""
        # Set participant and nonparticipating clients
        loaded_config = Config()

        ## 1. Initialize participanting clients
        participating_clients_ratio = (
            loaded_config.algorithm.personalization.participating_clients_ratio
            if hasattr(
                loaded_config.algorithm.personalization, "participating_clients_ratio"
            )
            else 1.0
        )

        #  clients will / will not participant in federated training
        self.number_participating_clients = int(
            self.total_clients * participating_clients_ratio
        )

        logging.info(
            fonts.colourize(
                "[%s] Total clients (%d), participanting clients (%d), "
                "participanting ratio (%.3f).",
                colour="blue",
            ),
            self,
            self.total_clients,
            self.number_participating_clients,
            participating_clients_ratio,
        )

    def set_various_clients_pool(self, clients_pool: List[int]):
        """Set various clients pool utilized in federated learning.

        Note: the participating clients pool will be set in the first round and no
            modification is performed afterwards.
        """
        
        # Only need to set the clients pool when they are empty.
        if self.participating_clients_pool is None:
            self.participating_clients_pool = clients_pool[
                : self.number_participating_clients
            ]

            logging.info(
                fonts.colourize(
                    "[%s] Prepared participanting clients pool: %s", colour="blue"
                ),
                self,
                self.participating_clients_pool,
            )

    def perform_normal_training(self, clients_pool: List[int], clients_count: int):
        """Operations to guarantee general federated learning without personalization."""

        # reset `clients_per_round` to the predefined hyper-parameter
        self.clients_per_round = Config().clients.per_round

        # set the clients_pool to be participating_clients_pool
        clients_pool = self.participating_clients_pool
        clients_count = self.clients_per_round

        # by default, we run the general federated training
        # the clients pool should be participating clients
        assert clients_count <= len(self.participating_clients_pool)

        return clients_pool, clients_count

    def perform_final_personalization(
        self, clients_pool: List[int], clients_count: int
    ):
        """Performing personalization after the final round."""
        logging.info(
            fonts.colourize(
                "Starting personalization after the final round.", colour="blue"
            )
        )
        # do personalization on all clients
        clients_pool = self.clients_pool

        # set clients for personalization
        self.clients_per_round = len(clients_pool)
        clients_count = self.clients_per_round

        # to terminate the personalization afterwards
        self.personalization_started = True

        return clients_pool, clients_count

    def before_clients_sampling(
        self, clients_pool: List[int], clients_count: int, **kwargs
    ):
        """Determine clients pool and clients count before samling clients."""

        # perform normal training
        clients_pool, clients_count = self.perform_normal_training(
            clients_pool, clients_count
        )
        # perform personalization
        if self.current_round > Config().trainer.rounds:
            clients_pool, clients_count = self.perform_final_personalization(
                clients_pool, clients_count
            )

        return clients_pool, clients_count

    def choose_clients(self, clients_pool: List[int], clients_count: int):
        """Chooses a subset of the clients to participate in each round.

        In plato, this input `clients_pool` contains total clients
        id by default.
        """
        # set required clients pool when possible
        self.set_various_clients_pool(clients_pool)

        clients_pool, clients_count = self.before_clients_sampling(
            clients_pool, clients_count
        )

        random.setstate(self.prng_state)

        # Select clients randomly
        selected_clients = random.sample(clients_pool, clients_count)

        self.prng_state = random.getstate()
        if selected_clients == len(clients_pool):
            logging.info("[%s] Selected all %d clients", self, len(selected_clients))
        else:
            logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients

    async def wrap_up(self):
        """Wrapping up when each round of training is done."""
        self.save_to_checkpoint()

        if self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")

            if self.personalization_started:
                logging.info(
                    fonts.colourize(
                        "Personalization completed after the final round.", colour="blue"
                    ),
                )
                await self._close()
