"""Unit tests for the personalization."""
import os
import unittest

os.environ['config_file'] = 'tests/TestsConfig/personalized_config.yml'

from plato.config import Config
import plato.models.registry as models_registry
from plato.trainers import optimizers
from plato.trainers import lr_schedulers


class LrSchedulerTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        __ = Config()

        print(Config().trainer)

        self.model = models_registry.get()
        self.optimizer = optimizers.get(self.model)
        self.lrs = optimizers.get(self.model)

    def assert_unequal_lrs_config(self, customize_lr):
        #self.assertEqual()
        pass

    def assert_unequal_optimizers(self, customize_optimizer):
        #self.assertEqual()
        pass

    def test_personalized_config(self):

        personalized_optimizer = optimizers.get(
            self.model,
            optimizer_name=Config().trainer.pers_optimizer,
            optim_params=Config().parameters.pers_optimizer._asdict())
        personalized_lrs = lr_schedulers.get(
            personalized_optimizer,
            10,
            lr_scheduler=Config().trainer.pers_lr_scheduler,
            lr_params=Config().parameters.pers_learning_rate._asdict())

        print(personalized_optimizer)
        print(personalized_lrs)


if __name__ == "__main__":
    unittest.main()
