"""
A customized trainer for the federated unlearning baseline clustering algorithm.

This trainer uses the composable trainer architecture with the strategy pattern,
implementing custom testing strategy for clustered model evaluation.
"""

import torch

from plato.config import Config
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.testing import DefaultTestingStrategy


class ClusteredTestingStrategy(DefaultTestingStrategy):
    """
    Testing strategy for evaluating multiple cluster models.

    This strategy extends the default testing strategy to support testing
    multiple cluster models on the same test dataset, which is required
    for the KNOT federated unlearning algorithm.
    """

    def test_clustered_models(
        self, testset, sampler, context, clustered_models, updated_cluster_ids
    ):
        """
        Separately perform model testing for all updated clusters.

        Args:
            testset: The test dataset
            sampler: Optional data sampler for the test set
            context: Training context with device, config, etc.
            clustered_models: dict mapping cluster IDs to model instances
            updated_cluster_ids: list/set of cluster IDs that were updated

        Returns:
            Dictionary mapping cluster IDs to test accuracy values
        """
        clustered_test_accuracy = {}
        config = (
            context.config if hasattr(context, "config") else Config().trainer._asdict()
        )
        batch_size = config.get("batch_size", 32)

        # Preparing the test data loader
        if sampler is None:
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False
            )
        else:
            test_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
            )

        for cluster_id in updated_cluster_ids:
            cluster_model = clustered_models[cluster_id]

            cluster_model.to(context.device)
            cluster_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for examples, labels in test_loader:
                    examples, labels = (
                        examples.to(context.device),
                        labels.to(context.device),
                    )

                    outputs = cluster_model(examples)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            cluster_acc = correct / total
            clustered_test_accuracy[cluster_id] = cluster_acc

        return clustered_test_accuracy


class Trainer(ComposableTrainer):
    """
    A federated learning trainer using the Knot algorithm.

    This trainer extends ComposableTrainer with a custom ClusteredTestingStrategy
    to support testing multiple cluster models. It uses the strategy pattern for
    all aspects of training and testing.

    The server can directly access the testing strategy via:
        trainer.testing_strategy.test_clustered_models(...)
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the Knot trainer with clustered testing strategy.

        Arguments:
            model: The model to train (class or instance)
            callbacks: List of callback classes or instances
        """
        # Initialize with custom clustered testing strategy
        super().__init__(
            model=model,
            callbacks=callbacks,
            testing_strategy=ClusteredTestingStrategy(),
        )
