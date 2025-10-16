"""
A personalized federated learning server that starts from a number of regular
rounds of federated learning. In these regular rounds, only a subset of the
total clients can be selected to perform the local update (the ratio of which is
a configuration setting). After all regular rounds are completed, it starts a
final round of personalization, where a selected subset of clients perform local
training using their local dataset.
"""

from plato.config import Config
from plato.servers import fedavg
from plato.servers.strategies.client_selection import (
    PersonalizedRatioSelectionStrategy,
)


class Server(fedavg.Server):
    """
    A personalzed FL server that controls how many clients will participate in
    the training process, and that adds a final personalization round with all
    clients sampled.
    """

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        aggregation_strategy=None,
        client_selection_strategy=None,
    ):
        ratio = 1.0
        if hasattr(Config().algorithm, "personalization"):
            personalization_cfg = Config().algorithm.personalization
            if hasattr(personalization_cfg, "participating_client_ratio"):
                ratio = personalization_cfg.participating_client_ratio

        personalization_rounds = Config().trainer.rounds

        if client_selection_strategy is None:
            client_selection_strategy = PersonalizedRatioSelectionStrategy(
                ratio=ratio,
                personalization_rounds=personalization_rounds,
            )

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )
        # Personalization starts after the final regular round of training
        self.personalization_started = False
        self.personalization_rounds = personalization_rounds

    async def wrap_up(self) -> None:
        """Wraps up when each round of training is done."""
        if self.personalization_started:
            await super().wrap_up()
        else:
            # If the target number of training rounds has been reached, start
            # the final round of training for personalization on the clients
            self.save_to_checkpoint()

            if self.current_round >= self.personalization_rounds:
                self.personalization_started = True
