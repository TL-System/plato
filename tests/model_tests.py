"""Unit tests for the model definition."""

import os
import unittest

os.environ["config_file"] = "TestsConfig/models_config.yml"

from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import optimizers


class ModelsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        __ = Config()

        # 1. define the main model to be the encoder
        self.model = models_registry.get()
        self.optimizer = optimizers.get(self.model)

    def test_model_config(self):
        """Test whether the models are defined based on the configuration files."""
        # test the defined models based on the
        # hyper-parameters

        # 1. whether the defined model is the encoder
        self.assertEqual(self.model.encoding_dim, 120)


if __name__ == "__main__":
    unittest.main()
