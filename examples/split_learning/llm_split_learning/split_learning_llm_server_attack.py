"""
A dishonest server which will try to attack and reconstruct user private data
    with the received intermediate features.
"""
from plato.servers import split_learning as split_learning_server
from plato.config import Config


class DishonestServer(split_learning_server.Server):
    """
    A dishonest server will decide whether to attack based on the given attacking interval.
    """

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate weight updates from the clients or train the model."""
        await super().aggregate_weights(updates, baseline_weights, weights_received)
        update = updates[0]
        report = update.report
        if (
            report.type == "features"
            and self.current_round % Config().parameters.attack.interval == 0
        ):
            # probably add some concurrency flag
            intermediate_features, __ = update
            reconstructed_words = self.trainer.attack(intermediate_features)
