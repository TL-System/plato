"""
A base server to perform personalized federated learning.

With this server, there will be two training processes. 

The first process performs conventional federated training. 
Besides, during the federated training process, only a subset of the total clients can be 
selected to perform the local update, while others will not be used. We call this subset of 
clients the participating clients. The number of participating clients is determined by the
`participating_client_ratio` in the configuration file. The default value is 1.0.

The second process performs personalization after the final round. The server will 
send the trained global model to clients so that each client can fine-tune the received 
global model based on local samples for multiple epochs to get the personalized models.

"""

from typing import List
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

        # A flag to indicate whether the personalization process has started
        # The personalization starts after the final communication round of
        # federated training (FL) has been completed
        self.personalization_started = False

    def choose_clients(self, clients_pool: List[int], clients_count: int):
        """Choose a subset of the clients to participate in each round."""
        # Start personalization as the current round exceeds the total rounds
        # set in the configuration file
        if self.current_round > Config().trainer.rounds:
            self.personalization_started = True
            # Choose `clients_count` from all clients to participate in the
            # personalization process
            return super().choose_clients(clients_pool, clients_count)
        else:
            # Get the `participating_client_ratio` from the configuration file
            # See more details in the comment at the top of this file
            participating_client_ratio = (
                Config().algorithm.personalization.participating_client_ratio
                if hasattr(
                    Config().algorithm.personalization, "participating_client_ratio"
                )
                else 1.0
            )
            # Choose `clients_count` from the participating clients, which is a subset
            # of total clients, to perform the local update
            return super().choose_clients(
                clients_pool[: int(self.total_clients * participating_client_ratio)],
                clients_count,
            )

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
