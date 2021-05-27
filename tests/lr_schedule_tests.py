"""Unit tests for the learning rate scheduler."""
import unittest
import warnings
from collections import namedtuple
import numpy as np

from plato.utils import optimizers
from plato.config import Config
import plato.models.registry as models_registry


class LrSchedulerTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        __ = Config()

        fields = [
            'optimizer', 'lr_schedule', 'learning_rate', 'momentum',
            'weight_decay', 'lr_gamma', 'lr_milestone_steps',
            'lr_warmup_steps', 'model_name'
        ]
        params = ['SGD', 'LambdaLR', 0.1, 0.5, 0.0, 0.0, '', '', 'resnet_18']
        Config().trainer = namedtuple('trainer', fields)(*params)

        self.model = models_registry.get()
        self.optimizer = optimizers.get_optimizer(self.model)

    def assertLrEquals(self, lr):
        self.assertEqual(np.round(self.optimizer.param_groups[0]['lr'], 10),
                         np.round(lr, 10))

    def test_vanilla(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            fields = [
                'optimizer', 'lr_schedule', 'learning_rate', 'momentum',
                'weight_decay', 'lr_gamma'
            ]
            params = ['SGD', 'LambdaLR', 0.1, 0.5, 0.0, 0.0]
            Config().trainer = namedtuple('trainer', fields)(*params)

            lrs = optimizers.get_lr_schedule(self.optimizer, 10)
            self.assertLrEquals(0.1)
            for _ in range(100):
                lrs.step()
            self.assertLrEquals(0.1)
            self.assertLrEquals(0.1)

    def test_milestones(self):
        self.assertLrEquals(0.1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            fields = [
                'optimizer', 'lr_schedule', 'learning_rate', 'momentum',
                'weight_decay', 'lr_gamma', 'lr_milestone_steps'
            ]
            params = ['SGD', 'LambdaLR', 0.1, 0.5, 0.0, 0.1, '2ep,4ep,7ep,8ep']

            Config().trainer = namedtuple('trainer', fields)(*params)
            self.assertLrEquals(0.1)

            lrs = optimizers.get_lr_schedule(self.optimizer, 10)

            self.assertLrEquals(0.1)
            for _ in range(19):
                lrs.step()
            self.assertLrEquals(1e-1)

            for _ in range(1):
                lrs.step()
            self.assertLrEquals(1e-2)
            for _ in range(19):
                lrs.step()
            self.assertLrEquals(1e-2)

            for _ in range(1):
                lrs.step()
            self.assertLrEquals(1e-3)
            for _ in range(29):
                lrs.step()
            self.assertLrEquals(1e-3)

            for _ in range(1):
                lrs.step()
            self.assertLrEquals(1e-4)
            for _ in range(9):
                lrs.step()
            self.assertLrEquals(1e-4)

            for _ in range(1):
                lrs.step()
            self.assertLrEquals(1e-5)
            for _ in range(100):
                lrs.step()
            self.assertLrEquals(1e-5)

    def test_warmup(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            fields = [
                'optimizer', 'lr_schedule', 'learning_rate', 'momentum',
                'weight_decay', 'lr_gamma', 'lr_warmup_steps'
            ]
            params = ['SGD', 'LambdaLR', 0.1, 0.5, 0.0, 0.0, '20it']

            Config().trainer = namedtuple('trainer', fields)(*params)

            lrs = optimizers.get_lr_schedule(self.optimizer, 10)

            for i in range(20):
                self.assertLrEquals(i / 20 * 0.1)
                lrs.step()
            self.assertLrEquals(0.1)
            for i in range(100):
                lrs.step()
                self.assertLrEquals(0.1)


if __name__ == '__main__':
    unittest.main()
