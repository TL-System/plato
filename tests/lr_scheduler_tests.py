"""Unit tests for the learning rate scheduler."""

import os
import unittest
import warnings
from collections import namedtuple

os.environ["config_file"] = "TestsConfig/fedavg_tests.yml"

import numpy as np

import plato.models.registry as models_registry
from plato.config import Config
from plato.trainers import lr_schedulers, optimizers


class LrSchedulerTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        __ = Config()

        fields = [
            "optimizer",
            "lr_scheduler",
            "model_name",
        ]
        params = ["SGD", "LambdaLR", "resnet_18"]
        Config().trainer = namedtuple("trainer", fields)(*params)

        fields = ["optimizer", "learning_rate"]
        Config().parameters = namedtuple("parameters", fields)

        fields = [
            "lr",
            "momentum",
            "weight_decay",
        ]
        params = [0.1, 0.5, 0.0]
        Config().parameters.optimizer = namedtuple("optimizer", fields)(*params)

        fields = [
            "gamma",
            "milestone_steps",
            "warmup_steps",
        ]
        params = [0.0, "", ""]
        Config().parameters.learning_rate = namedtuple("learning_rate", fields)(*params)

        self.model = models_registry.get()
        self.optimizer = optimizers.get(self.model)

    def assert_lr_equal(self, lr):
        self.assertEqual(
            np.round(self.optimizer.param_groups[0]["lr"], 10), np.round(lr, 10)
        )

    def test_vanilla(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            fields = [
                "optimizer",
                "lr_scheduler",
            ]
            params = ["SGD", "LambdaLR"]
            Config().trainer = namedtuple("trainer", fields)(*params)

            fields = ["optimizer", "learning_rate"]
            Config().parameters = namedtuple("parameters", fields)

            fields = [
                "lr",
                "momentum",
                "weight_decay",
            ]
            params = [0.1, 0.5, 0.0]
            Config().parameters.optimizer = namedtuple("optimizer", fields)(*params)

            fields = [
                "gamma",
            ]
            params = [0.0]
            Config().parameters.learning_rate = namedtuple("learning_rate", fields)(
                *params
            )

            lrs = lr_schedulers.get(self.optimizer, 10)

            self.assert_lr_equal(0.1)
            for _ in range(100):
                lrs.step()
            self.assert_lr_equal(0.1)
            self.assert_lr_equal(0.1)

    def test_milestones(self):
        self.assert_lr_equal(0.1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            fields = [
                "optimizer",
                "lr_scheduler",
            ]
            params = ["SGD", "LambdaLR"]
            Config().trainer = namedtuple("trainer", fields)(*params)

            fields = ["optimizer", "learning_rate"]
            Config().parameters = namedtuple("parameters", fields)

            fields = [
                "lr",
                "momentum",
                "weight_decay",
            ]
            params = [0.1, 0.5, 0.0]
            Config().parameters.optimizer = namedtuple("optimizer", fields)(*params)

            fields = [
                "gamma",
                "milestone_steps",
            ]
            params = [0.1, "2ep,4ep,7ep,8ep"]
            Config().parameters.learning_rate = namedtuple("learning_rate", fields)(
                *params
            )

            self.assert_lr_equal(0.1)

            lrs = lr_schedulers.get(self.optimizer, 10)

            self.assert_lr_equal(0.1)
            for _ in range(19):
                lrs.step()
            self.assert_lr_equal(1e-1)

            for _ in range(1):
                lrs.step()
            self.assert_lr_equal(1e-2)
            for _ in range(19):
                lrs.step()
            self.assert_lr_equal(1e-2)

            for _ in range(1):
                lrs.step()
            self.assert_lr_equal(1e-3)
            for _ in range(29):
                lrs.step()
            self.assert_lr_equal(1e-3)

            for _ in range(1):
                lrs.step()
            self.assert_lr_equal(1e-4)
            for _ in range(9):
                lrs.step()
            self.assert_lr_equal(1e-4)

            for _ in range(1):
                lrs.step()
            self.assert_lr_equal(1e-5)
            for _ in range(100):
                lrs.step()
            self.assert_lr_equal(1e-5)

    def test_warmup(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            fields = [
                "optimizer",
                "lr_scheduler",
            ]
            params = ["SGD", "LambdaLR"]
            Config().trainer = namedtuple("trainer", fields)(*params)

            fields = ["optimizer", "learning_rate"]
            Config().parameters = namedtuple("parameters", fields)

            fields = [
                "lr",
                "momentum",
                "weight_decay",
            ]
            params = [0.1, 0.5, 0.0]
            Config().parameters.optimizer = namedtuple("optimizer", fields)(*params)

            fields = [
                "gamma",
                "warmup_steps",
            ]
            params = [0.1, "20it"]
            Config().parameters.learning_rate = namedtuple("learning_rate", fields)(
                *params
            )

            lrs = lr_schedulers.get(self.optimizer, 10)

            for i in range(20):
                self.assert_lr_equal(i / 20 * 0.1)
                lrs.step()
            self.assert_lr_equal(0.1)
            for i in range(100):
                lrs.step()
                self.assert_lr_equal(0.1)


if __name__ == "__main__":
    unittest.main()
