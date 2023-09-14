"""
A customized trainer for the federated unlearning baseline clustering algorithm.

"""
import math

from transformers import Trainer as HuggingFaceTrainer
from transformers import default_data_collator

from plato.config import Config
from plato.trainers import huggingface


class Trainer(huggingface.Trainer):
    """A federated learning trainer using the Knot algorithm."""

    async def server_clustered_test(self, testset, sampler=None, **kwargs):
        """ Separately perfrom the model test for all clutsers. """
        # The models within each cluster should be provided in the argument,
        # and it should be a dictionary in which the keys are cluster IDs,
        # and the values are the corresponding models
        assert "clustered_models" in kwargs

        # Which clusters have been updated in this aggregation should be provided
        # as either a list or a set
        assert "updated_cluster_ids" in kwargs

        clustered_models = kwargs["clustered_models"]
        updated_cluster_ids = kwargs["updated_cluster_ids"]

        clustered_test_accuracy = {}

        for cluster_id in updated_cluster_ids:
            cluster_model = clustered_models[cluster_id]

            self.trainer = HuggingFaceTrainer(
                model=cluster_model,
                args=self.training_args,
                train_dataset=None,
                eval_dataset=testset,
                tokenizer=self.tokenizer,
                data_collator=default_data_collator)

            metrics = self.trainer.evaluate()

            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            clustered_test_accuracy[cluster_id] = perplexity

        return clustered_test_accuracy
