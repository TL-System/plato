"""Unit tests for the learning rate scheduler."""
import os
import sys
import unittest
import warnings
from collections import namedtuple
import numpy as np

# To import modules from the parent directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from training import optimizers
import models.registry as models_registry
from config import Config


class LrSchedulerTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        __ = Config()

        fields = [
            'optimizer', 'learning_rate', 'momentum', 'weight_decay',
            'lr_gamma', 'lr_milestone_steps', 'lr_warmup_steps'
        ]
        params = ['SGD', 0.1, 0.5, 0.0, 0.0, '', '']
        Config().training = namedtuple('training', fields)(*params)

        self.model = models_registry.get('cifar_resnet_18')
        self.optimizer = optimizers.get_optimizer(self.model)

    def assertLrEquals(self, lr):
        self.assertEqual(np.round(self.optimizer.param_groups[0]['lr'], 10),
                         np.round(lr, 10))

    def test_vanilla(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            fields = [
                'optimizer', 'learning_rate', 'momentum', 'weight_decay',
                'lr_gamma', 'lr_milestone_steps', 'lr_warmup_steps'
            ]
            params = ['SGD', 0.1, 0.5, 0.0, 0.0, '', '']
            Config().training = namedtuple('training', fields)(*params)

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
                'optimizer', 'learning_rate', 'momentum', 'weight_decay',
                'lr_gamma', 'lr_milestone_steps', 'lr_warmup_steps'
            ]
            params = ['SGD', 0.1, 0.5, 0.0, 0.1, '2ep,4ep,7ep,8ep', '']

            Config().training = namedtuple('training', fields)(*params)
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
                'optimizer', 'learning_rate', 'momentum', 'weight_decay',
                'lr_gamma', 'lr_milestone_steps', 'lr_warmup_steps'
            ]
            params = ['SGD', 0.1, 0.5, 0.0, 0.0, '', '20it']

            Config().training = namedtuple('training', fields)(*params)

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
