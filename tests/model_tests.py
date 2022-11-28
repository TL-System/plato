"""Unit tests for the model definition."""
import os
import unittest

os.environ["config_file"] = "tests/TestsConfig/models_config.yml"

from plato.models import registry as models_registry
from plato.config import Config
from plato.trainers import optimizers


class ModelsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        __ = Config()

        # 1. define the main model to be the encoder
        self.model = models_registry.get()
        self.optimizer = optimizers.get(self.model)

        # 2. define the personalized model to be normal resnet
        self.personalized_model = models_registry.get(
            model_type=Config().trainer.personalized_model_type,
            model_name=Config().trainer.personalized_model_name,
            model_params=Config().parameters.personalized_model._asdict(),
        )
        self.personalized_optimizer = optimizers.get(
            self.personalized_model,
            optimizer_name=Config().trainer.personalized_optimizer,
            optim_params=Config().parameters.personalized_optimizer._asdict(),
        )

    def test_model_config(self):
        """ Test whether the models are defined based on the configuration files. """
        # test the defined models based on the
        # hyper-parameters

        # 1. whether the defined model is the encoder
        self.assertEqual(self.model.encoding_dim, 120)

        # 2. test whether the defined personalized model is
        # a normal resnet with output's dimension 'num_classes'
        self.assertEqual(self.personalized_model.linear.out_features, 20)


if __name__ == "__main__":
    unittest.main()
