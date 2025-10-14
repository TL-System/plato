"""
A customized trainer for the federated unlearning baseline clustering algorithm.

This trainer uses the strategy pattern with a custom testing strategy
for clustered model evaluation using HuggingFace transformers.
"""

import math

from transformers import Trainer as HuggingFaceTrainer
from transformers import default_data_collator

from plato.trainers import huggingface


class ClusteredHuggingFaceTestingStrategy:
    """
    Testing strategy for evaluating multiple cluster models using HuggingFace.

    This strategy provides clustered model testing for the KNOT federated
    unlearning algorithm using HuggingFace's transformer models.
    """

    def __init__(self, training_args, tokenizer):
        """
        Initialize the clustered testing strategy.

        Args:
            training_args: HuggingFace TrainingArguments
            tokenizer: HuggingFace tokenizer
        """
        self.training_args = training_args
        self.tokenizer = tokenizer

    def test_clustered_models(
        self, testset, sampler, clustered_models, updated_cluster_ids
    ):
        """
        Separately perform model testing for all updated clusters.

        Args:
            testset: The test dataset
            sampler: Optional data sampler for the test set
            clustered_models: dict mapping cluster IDs to model instances
            updated_cluster_ids: list/set of cluster IDs that were updated

        Returns:
            Dictionary mapping cluster IDs to perplexity values
        """
        clustered_test_accuracy = {}

        for cluster_id in updated_cluster_ids:
            cluster_model = clustered_models[cluster_id]

            trainer = HuggingFaceTrainer(
                model=cluster_model,
                args=self.training_args,
                train_dataset=None,
                eval_dataset=testset,
                tokenizer=self.tokenizer,
                data_collator=default_data_collator,
            )

            metrics = trainer.evaluate()

            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            clustered_test_accuracy[cluster_id] = perplexity

        return clustered_test_accuracy


class Trainer(huggingface.Trainer):
    """
    A federated learning trainer using the Knot algorithm with HuggingFace models.

    This trainer extends the HuggingFace trainer with a custom testing strategy
    for evaluating multiple cluster models.

    The server can directly access the testing strategy via:
        trainer.testing_strategy.test_clustered_models(...)
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the KNOT HuggingFace trainer.

        Args:
            model: The model to train (HuggingFace model)
            callbacks: List of callback classes or instances
        """
        super().__init__(model=model, callbacks=callbacks)

        # Initialize the clustered testing strategy
        self.testing_strategy = ClusteredHuggingFaceTestingStrategy(
            self.training_args, self.tokenizer
        )
