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
        self.rouge = {}
        self.attack_accuracy = 0.0
        self.attack_started = False

    def attack(self, update):
        """
        Conduct the attack in the background. Otherwise, the client will notice
            as the aggregation time is abnormally long.
        """
        self.attack_started = True
        intermediate_features, labels = update
        evaluation_metrics = self.trainer.attack(intermediate_features, labels)
        self.rouge = evaluation_metrics["ROUGE"]
        self.attack_accuracy = evaluation_metrics["attack_accuracy"]
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
            # proc = mp.Process(
            #     target=self.attack,
            #     args=(update),
            # )
            # proc.start()
            self.attack(update)

    def get_logged_items(self):
        """Get items to be logged by the LogProgressCallback class in a .csv file."""
        logged_items = super().get_logged_items()
        logged_items["attack_accuracy"] = self.attack_accuracy
        logged_items["rouge1_fm"] = self.rouge["rouge1_fmeasure"].item()
        logged_items["rouge1_p"] = self.rouge["rouge1_precision"].item()
        logged_items["rouge1_r"] = self.rouge["rouge1_recall"].item()
        logged_items["rouge2_fm"] = self.rouge["rouge2_fmeasure"].item()
        logged_items["rouge2_p"] = self.rouge["rouge2_precision"].item()
        logged_items["rouge2_r"] = self.rouge["rouge2_recall"].item()
        logged_items["rougeL_fm"] = self.rouge["rougeL_fmeasure"].item()
        logged_items["rougeL_p"] = self.rouge["rougeL_precision"].item()
        logged_items["rougeL_r"] = self.rouge["rougeL_recall"].item()
        logged_items["rougeLsum_fm"] = self.rouge["rougeLsum_fmeasure"].item()
        logged_items["rougeLsum_p"] = self.rouge["rougeLsum_precision"].item()
        logged_items["rougeLsum_r"] = self.rouge["rougeLsum_recall"].item()

        return logged_items
