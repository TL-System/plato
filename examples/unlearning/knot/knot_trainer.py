"""
A customized trainer for the federated unlearning baseline clustering algorithm.

"""

import asyncio

import torch

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """A federated learning trainer using the Knot algorithm."""

    def server_clustered_test(self, testset, sampler=None, **kwargs):
        """Separately perform the model test for all clusters."""
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
        config = Config().trainer._asdict()

        # Preparing the test data loader
        if sampler is None:
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=config["batch_size"], shuffle=False
            )
        else:
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=config["batch_size"], shuffle=False, sampler=sampler
            )

        for cluster_id in updated_cluster_ids:
            cluster_model = clustered_models[cluster_id]

            cluster_model.to(self.device)
            cluster_model.eval()

            correct = 0
            total = 0
            with torch.no_grad():
                for examples, labels in test_loader:
                    examples, labels = examples.to(self.device), labels.to(self.device)

                    outputs = cluster_model(examples)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            cluster_acc = correct / total
            clustered_test_accuracy[cluster_id] = cluster_acc

        return clustered_test_accuracy
