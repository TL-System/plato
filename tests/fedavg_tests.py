import os
import sys
import copy
import unittest
import numpy as np
import torch

# To import modules from the parent directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models.base import Model
from clients import simple
from algorithms import fedavg as fedavg_alg
from servers import fedavg as fedavg_server
from trainers import basic


class InnerProductModel(Model):
    @staticmethod
    def is_valid_model_type(model_type):
        raise NotImplementedError

    @staticmethod
    def get_model_from_type(model_type):
        raise NotImplementedError

    @property
    def loss_criterion(self):
        return torch.nn.MSELoss()

    def __init__(self, n):
        super().__init__()
        self.layer = torch.nn.Linear(n, 1, bias=False)
        self.layer.weight.data = torch.arange(n, dtype=torch.float32)

    def forward(self, x):
        return self.layer(x)


class FedAvgTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.model = InnerProductModel(10)
        self.example = torch.ones(1, 10)
        self.label = torch.ones(1) * 40.0
        self.trainer = basic.Trainer(model=self.model)
        self.algorithm = fedavg_alg.Algorithm(trainer=self.trainer)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def test_forward(self):
        self.assertIsNotNone(self.model)
        weights = self.algorithm.extract_weights()
        print("\nTesting forward pass.")
        print(f"Weights: {weights}")
        self.assertEqual(45.0, self.model(self.example).item())

    def test_backward(self):
        print("\nTesting backward pass.")
        self.model.train()

        self.optimizer.zero_grad()
        self.model.loss_criterion(self.model(self.example),
                                  self.label).backward()
        self.optimizer.step()
        self.assertEqual(44.0, self.model(self.example).item())
        weights = self.algorithm.extract_weights()
        print(f"Weights: {weights}")

        self.optimizer.zero_grad()
        self.model.loss_criterion(self.model(self.example),
                                  self.label).backward()
        self.optimizer.step()
        self.assertEqual(43.2, np.round(self.model(self.example).item(), 4))
        weights = self.algorithm.extract_weights()
        print(f"Weights: {weights}")

        self.optimizer.zero_grad()
        self.model.loss_criterion(self.model(self.example),
                                  self.label).backward()
        self.optimizer.step()
        self.assertEqual(42.56, np.round(self.model(self.example).item(), 4))
        weights = self.algorithm.extract_weights()
        print(f"Weights: {weights}")

    def test_fedavg_aggregation(self):
        print("\nTesting federated averaging.")
        reports = []
        model = copy.deepcopy(self.model)
        server = fedavg_server.Server(model=model)
        trainer = basic.Trainer(model=model)
        algorithm = fedavg_alg.Algorithm(trainer=trainer)
        server.trainer = trainer      
        server.algorithm = algorithm

        weights = copy.deepcopy(self.algorithm.extract_weights())
        print(f"Report 1 weights: {weights}")
        reports.append((simple.Report(1, 100, 0, 0), weights))

        self.model.train()

        self.optimizer.zero_grad()
        self.model.loss_criterion(self.model(self.example),
                                  self.label).backward()
        self.optimizer.step()
        self.assertEqual(44.0, self.model(self.example).item())
        weights = copy.deepcopy(self.algorithm.extract_weights())
        print(f"Report 2 weights: {weights}")
        reports.append((simple.Report(1, 100, 0, 0), weights))

        self.optimizer.zero_grad()
        self.model.loss_criterion(self.model(self.example),
                                  self.label).backward()
        self.optimizer.step()
        self.assertEqual(43.2, np.round(self.model(self.example).item(), 4))
        weights = copy.deepcopy(self.algorithm.extract_weights())
        print(f"Report 3 Weights: {weights}")
        reports.append((simple.Report(1, 100, 0, 0), weights))

        self.optimizer.zero_grad()
        self.model.loss_criterion(self.model(self.example),
                                  self.label).backward()
        self.optimizer.step()
        self.assertEqual(42.56, np.round(self.model(self.example).item(), 4))
        weights = copy.deepcopy(self.algorithm.extract_weights())
        print(f"Report 4 Weights: {weights}")
        reports.append((simple.Report(1, 100, 0, 0), weights))

        print(
            f"Weights before federated averaging: {server.model.layer.weight.data}"
        )
        updated_weights = server.federated_averaging(reports)
        server.algorithm.load_weights(updated_weights)
        print(
            f"Weights after federated averaging: {server.model.layer.weight.data}"
        )
        self.assertEqual(42.56, np.round(self.model(self.example).item(), 4))


if __name__ == '__main__':
    unittest.main()
