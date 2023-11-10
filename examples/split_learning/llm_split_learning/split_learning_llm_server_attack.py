"""
A dishonest server which will try to attack and reconstruct user private data
    with the received intermediate features.
"""
import multiprocessing as mp
from plato.servers import split_learning as split_learning_server
from plato.config import Config


class DishonestServer(split_learning_server.Server):
    """
    A dishonest server will decide whether to attack based on the given attacking interval.
    """

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.bleu_score = 0.0
        self.rouge_score = 0.0
        self.attack_accuracy = 0.0
        self.attack_started = False

    def attack(self, update):
        """
        Conduct the attack in the background. Otherwise, the client will notice
            as the aggregation time is abnormally long.
        """
        self.attack_started = True
        intermediate_features, labels = update
        reconstructed_token_ids = self.trainer.attack(intermediate_features)
        evaluation_metrics = self.trainer.evaluate_attack(
            reconstructed_token_ids, labels
        )
        self.bleu_score = evaluation_metrics["BLEU"]
        self.rouge_score = evaluation_metrics["ROGUE"]
        self.attack_accuracy = evaluation_metrics["attack accuracy"]
        self.attack_started = False

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate weight updates from the clients or train the model."""
        await super().aggregate_weights(updates, baseline_weights, weights_received)
        update = updates[0]
        report = update.report
        if (
            report.type == "features"
            and self.current_round % Config().parameters.attack.interval == 0
            and not self.attack_started
        ):
            # probably add some concurrency flag
            proc = mp.Process(
                target=self.attack,
                args=(update),
            )
            proc.start()

    def get_logged_items(self):
        """Get items to be logged by the LogProgressCallback class in a .csv file."""
        logged_items = super().get_logged_items()
        logged_items["BLEU"] = self.bleu_score
        logged_items["ROGUE"] = self.rouge_score
        logged_items["attack accuracy"] = self.attack_accuracy

        return logged_items
