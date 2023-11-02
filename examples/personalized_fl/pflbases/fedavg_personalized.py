"""
A personalized federated learning server that starts from a number of regular
rounds of federated learning. In these regular rounds, only a subset of the
total clients can be selected to perform the local update (the ratio of which is
a configuration setting). After all regular rounds are completed, it starts a
final round of personalization, where a selected subset of clients perform local
training using their local dataset.
"""

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """
    A personalzed FL server that controls how many clients will participate in
    the training process, and that adds a final personalization round with all
    clients sampled.
    """

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
        # Personalization starts after the final regular round of training
        self.personalization_started = False

    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        if self.current_round > Config().trainer.rounds:
            # In the final personalization round, choose from all clients
            return super().choose_clients(clients_pool, clients_count)
        else:
            ratio = Config().algorithm.personalization.participating_client_ratio

            return super().choose_clients(
                clients_pool[: int(self.total_clients * ratio)],
                clients_count,
            )

    async def wrap_up(self) -> None:
        """Wraps up when each round of training is done."""
        if self.personalization_started:
            await super().wrap_up()
        else:
            # If the target number of training rounds has been reached, start
            # the final round of training for personalization on the clients
            self.save_to_checkpoint()

            if self.current_round >= Config().trainer.rounds:
                self.personalization_started = True
