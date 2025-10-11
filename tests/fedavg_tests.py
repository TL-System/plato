import asyncio
import copy
import os
import unittest

import numpy as np
import torch

os.environ["config_file"] = "TestsConfig/fedavg_tests.yml"


from plato.algorithms import registry as algorithms_registry
from plato.clients import simple
from plato.config import Config
from plato.servers import fedavg as fedavg_server
from plato.trainers import basic


class InnerProductModel(torch.nn.Module):
    def __init__(self, n: int = 10):
        super().__init__()
        self.layer = torch.nn.Linear(n, 1, bias=False)
        self.layer.weight.data = torch.arange(n, dtype=torch.float32)

        self.head = torch.nn.Linear(1, 1, bias=False)
        self.head.weight.data = torch.arange(1, 2, dtype=torch.float32)

    @staticmethod
    def is_valid_model_type(model_type):
        raise NotImplementedError

    @staticmethod
    def get_model_from_type(model_type):
        raise NotImplementedError

    @property
    def loss_criterion(self):
        return torch.nn.MSELoss()

    def forward(self, x):
        return self.layer(x)


async def test_fedavg_aggregation(self):
    """Testing the federated averaging implementation."""

    print("\nTesting federated averaging.")
    updates = []
    model = InnerProductModel

    trainer = basic.Trainer
    algorithm = algorithms_registry.registered_algorithms[Config().algorithm.type]
    server = fedavg_server.Server(model=model, algorithm=algorithm, trainer=trainer)
    server.init_trainer()

    weights = copy.deepcopy(self.algorithm.extract_weights())
    print(f"Report 1 weights: {weights}")
    updates.append(
        simple.SimpleNamespace(
            client_id=1,
            report=simple.SimpleNamespace(
                client_id=1,
                num_samples=100,
                accuracy=0,
                training_time=0,
                comm_time=0,
                update_response=False,
            ),
            payload=weights,
            staleness=0,
        )
    )

    self.trainer.model.train()

    self.optimizer.zero_grad()
    self.trainer.model.loss_criterion(
        self.trainer.model(self.example), self.label
    ).backward()
    self.optimizer.step()
    self.trainer.model.head.weight.data -= 0.1

    self.assertEqual(44.0, self.trainer.model(self.example).item())
    weights = copy.deepcopy(self.algorithm.extract_weights())
    print(f"Report 2 weights: {weights}")
    updates.append(
        simple.SimpleNamespace(
            client_id=2,
            report=simple.SimpleNamespace(
                client_id=2,
                num_samples=100,
                accuracy=0,
                training_time=0,
                comm_time=0,
                update_response=False,
            ),
            payload=weights,
            staleness=0,
        )
    )

    self.optimizer.zero_grad()
    self.trainer.model.loss_criterion(
        self.trainer.model(self.example), self.label
    ).backward()
    self.optimizer.step()
    self.trainer.model.head.weight.data -= 0.1
    self.assertEqual(43.2, np.round(self.trainer.model(self.example).item(), 4))
    weights = copy.deepcopy(self.algorithm.extract_weights())
    print(f"Report 3 Weights: {weights}")
    updates.append(
        simple.SimpleNamespace(
            client_id=3,
            report=simple.SimpleNamespace(
                client_id=3,
                num_samples=100,
                accuracy=0,
                training_time=0,
                comm_time=0,
                update_response=False,
            ),
            payload=weights,
            staleness=0,
        )
    )

    self.optimizer.zero_grad()
    self.trainer.model.loss_criterion(
        self.trainer.model(self.example), self.label
    ).backward()
    self.optimizer.step()
    self.trainer.model.head.weight.data -= 0.1

    self.assertEqual(42.56, np.round(self.trainer.model(self.example).item(), 4))
    weights = copy.deepcopy(self.algorithm.extract_weights())
    print(f"Report 4 Weights: {weights}")
    updates.append(
        simple.SimpleNamespace(
            client_id=4,
            report=simple.SimpleNamespace(
                client_id=4,
                num_samples=100,
                accuracy=0,
                training_time=0,
                comm_time=0,
                update_response=False,
            ),
            payload=weights,
            staleness=0,
        )
    )

    print(
        f"Weights of the layer before federated averaging: {server.trainer.model.layer.weight.data}"
    )
    print(
        f"Weights of the head before federated averaging: {server.trainer.model.head.weight.data}"
    )
    weights_received = [update.payload for update in updates]
    baseline_weights = server.algorithm.extract_weights()
    deltas_received = server.algorithm.compute_weight_deltas(
        baseline_weights, weights_received
    )
    deltas = await server.aggregate_deltas(updates, deltas_received)

    updated_weights = server.algorithm.update_weights(deltas)

    server.algorithm.load_weights(updated_weights)
    print(
        f"Weights of the layer after federated averaging: {server.trainer.model.layer.weight.data}"
    )
    print(
        f"Weights of the head after federated averaging: {server.trainer.model.head.weight.data}"
    )
    self.assertEqual(42.56, np.round(self.trainer.model(self.example).item(), 4))


class FedAvgTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.model = InnerProductModel
        self.example = torch.ones(1, 10)
        self.label = torch.ones(1) * 40.0
        self.trainer = basic.Trainer(model=self.model)
        self.algorithm = algorithms_registry.get(trainer=self.trainer)
        self.optimizer = torch.optim.SGD(self.trainer.model.parameters(), lr=0.01)

    def test_forward(self):
        self.assertIsNotNone(self.model)
        weights = self.algorithm.extract_weights()
        print("\nTesting forward pass.")
        print(f"Weights: {weights}")
        self.assertEqual(45.0, self.trainer.model(self.example).item())

    def test_backward(self):
        print("\nTesting backward pass.")
        self.trainer.model.train()
        if hasattr(self.algorithm, "extract_submodules_name"):
            print(
                "\nTesting submodules extraction.",
                self.algorithm.extract_submodules_name(
                    self.trainer.model.state_dict().keys()
                ),
            )
        if hasattr(self.algorithm, "is_consistent_weights"):
            print(
                "\nTesting weights consistency judge.",
                self.algorithm.is_consistent_weights(
                    self.trainer.model.state_dict().keys()
                ),
            )

        self.optimizer.zero_grad()
        self.trainer.model.loss_criterion(
            self.trainer.model(self.example), self.label
        ).backward()
        self.optimizer.step()

        self.assertEqual(44.0, self.trainer.model(self.example).item())
        weights = self.algorithm.extract_weights()
        print(f"Weights: {weights}")

        self.optimizer.zero_grad()
        self.trainer.model.loss_criterion(
            self.trainer.model(self.example), self.label
        ).backward()
        self.optimizer.step()
        self.assertEqual(43.2, np.round(self.trainer.model(self.example).item(), 4))
        weights = self.algorithm.extract_weights()
        print(f"Weights: {weights}")

        self.optimizer.zero_grad()
        self.trainer.model.loss_criterion(
            self.trainer.model(self.example), self.label
        ).backward()
        self.optimizer.step()
        self.assertEqual(42.56, np.round(self.trainer.model(self.example).item(), 4))
        weights = self.algorithm.extract_weights()
        print(f"Weights: {weights}")

    def test_fedavg_aggregation(self):
        asyncio.run(test_fedavg_aggregation(self))


if __name__ == "__main__":
    unittest.main()
