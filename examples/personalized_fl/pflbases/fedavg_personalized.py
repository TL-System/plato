"""
A base server to perform personalized federated learning.
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

        self.personalization_started = False

    def choose_clients(self, clients_pool: List[int], clients_count: int):
        """Choose a subset of the clients to participate in each round."""
        # Set required client pools when possible
        if self.current_round > Config().trainer.rounds:
            self.personalization_started = True
            return super().choose_clients(clients_pool, clients_count)
        else:
            participating_client_ratio = (
                Config().algorithm.personalization.participating_client_ratio
                if hasattr(
                    Config().algorithm.personalization, "participating_client_ratio"
                )
                else 1.0
            )
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
