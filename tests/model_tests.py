"""Unit tests for the personalization."""
import os
import unittest

os.environ["config_file"] = "tests/TestsConfig/models_config.yml"

from plato.models import registry as models_registry
from plato.config import Config
from plato.trainers import lr_schedulers, optimizers


class ModelsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        __ = Config()

        self.model = models_registry.get()
        self.optimizer = optimizers.get(self.model)

        self.personalized_model = models_registry.get(
            model_name=Config().trainer.personalized_model_name,
            model_params=Config().parameters.personalized_model._asdict(),
        )
        # 2. optimizer
        self.personalized_optimizer = optimizers.get(
            self.personalized_model,
            optimizer_name=Config().trainer.personalized_optimizer,
            optim_params=Config().parameters.personalized_optimizer._asdict(),
        )

    def test_model_config(self):
        # test the defined models

        print(self.model)
        print(self.personalized_model)
        # # test whether loading different hyper-parameters:
        # # 1. for the model
        # self.assertNotEqual(self.model.__str__(), personalized_model.__str__())
        # self.assertNotEqual(
        #     self.model.fc5.out_features, personalized_model.linear.out_features
        # )
        # # 2. for the optimizer.
        # self.assertNotEqual(
        #     self.optimizer.param_groups[0]["lr"],
        #     personalized_optimizer.param_groups[0]["lr"],
        # )
        # self.assertNotEqual(
        #     self.optimizer.param_groups[0]["weight_decay"],
        #     personalized_optimizer.param_groups[0]["weight_decay"],
        # )


if __name__ == "__main__":
    unittest.main()
