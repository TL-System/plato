"""
A curious server which will try to attack and reconstruct user private data
    with the received intermediate features.
"""

import multiprocessing as mp

from plato.config import Config
from plato.servers import split_learning as split_learning_server


class CuriousServer(split_learning_server.Server):
    """
    A curious server will decide whether to attack based on the given attacking interval.
    """

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.rouge = {
            "rouge1_fm": 0,
            "rouge1_p": 0,
            "rouge1_r": 0,
            "rouge2_fm": 0,
            "rouge2_p": 0,
            "rouge2_r": 0,
            "rougeL_fm": 0,
            "rougeL_p": 0,
            "rougeL_r": 0,
            "rougeLsum_fm": 0,
            "rougeLsum_p": 0,
            "rougeLsum_r": 0,
        }
        self.attack_accuracy = 0.0
        self.attack_started = False

    def attack(self, update):
        """
        The function to conduct the attack and update metrics for
            text generation to evaluate the attacks.
        """
        self.attack_started = True
        intermediate_features, labels = update[0]
        evaluation_metrics = self.trainer.attack(intermediate_features, labels)
        rouge_metrics = evaluation_metrics["ROUGE"]
        self.rouge["rouge1_fm"] = rouge_metrics["rouge1_fmeasure"].item()
        self.rouge["rouge1_p"] = rouge_metrics["rouge1_precision"].item()
        self.rouge["rouge1_r"] = rouge_metrics["rouge1_recall"].item()
        self.rouge["rouge2_fm"] = rouge_metrics["rouge2_fmeasure"].item()
        self.rouge["rouge2_p"] = rouge_metrics["rouge2_precision"].item()
        self.rouge["rouge2_r"] = rouge_metrics["rouge2_recall"].item()
        self.rouge["rougeL_fm"] = rouge_metrics["rougeL_fmeasure"].item()
        self.rouge["rougeL_p"] = rouge_metrics["rougeL_precision"].item()
        self.rouge["rougeL_r"] = rouge_metrics["rougeL_recall"].item()
        self.rouge["rougeLsum_fm"] = rouge_metrics["rougeLsum_fmeasure"].item()
        self.rouge["rougeLsum_p"] = rouge_metrics["rougeLsum_precision"].item()
        self.rouge["rougeLsum_r"] = rouge_metrics["rougeLsum_recall"].item()
        self.attack_accuracy = evaluation_metrics["attack_accuracy"]
        self.attack_started = False

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate weight updates from the clients or train the model."""
        updated_weights = await super().aggregate_weights(
            updates, baseline_weights, weights_received
        )
        update = updates[0]
        report = update.report
        if (
            report.type == "features"
            and self.current_round % Config().parameters.attack.interval == 0
            and not self.attack_started
        ):
            # Conduct the attack in the background. Otherwise, the client will notice
            #   as the aggregation time is abnormally long.
            # probably add some concurrency flag
            # proc = mp.Process(
            #     target=self.attack,
            #     args=(update),
            # )
            # proc.start()
            self.attack(update.payload)
        return updated_weights

    def get_logged_items(self):
        """Get items to be logged by the LogProgressCallback class in a .csv file."""
        logged_items = super().get_logged_items()
        logged_items["attack_accuracy"] = self.attack_accuracy
        logged_items["rouge1_fm"] = self.rouge["rouge1_fm"]
        logged_items["rouge1_p"] = self.rouge["rouge1_p"]
        logged_items["rouge1_r"] = self.rouge["rouge1_r"]
        logged_items["rouge2_fm"] = self.rouge["rouge2_fm"]
        logged_items["rouge2_p"] = self.rouge["rouge2_p"]
        logged_items["rouge2_r"] = self.rouge["rouge2_r"]
        logged_items["rougeL_fm"] = self.rouge["rougeL_fm"]
        logged_items["rougeL_p"] = self.rouge["rougeL_p"]
        logged_items["rougeL_r"] = self.rouge["rougeL_r"]
        logged_items["rougeLsum_fm"] = self.rouge["rougeLsum_fm"]
        logged_items["rougeLsum_p"] = self.rouge["rougeLsum_p"]
        logged_items["rougeLsum_r"] = self.rouge["rougeLsum_r"]

        return logged_items
