"""Unit tests for checkpoint operations."""

import logging
import os
import unittest

os.environ["config_file"] = "TestsConfig/models_config.yml"

from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import optimizers
from plato.utils import checkpoint_operator


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        __ = Config()

        # 1. define the main model to be the encoder
        self.model = models_registry.get()
        self.optimizer = optimizers.get(self.model)

    def test_checkpoint_saving(self):
        """Test operations for checkpoint saving."""
        ckp_operator = checkpoint_operator.CheckpointsOperator(
            checkpoints_dir="checkpoints/"
        )
        ckp_operator.save_checkpoint(
            self.model.state_dict(),
            checkpoints_name=["model.pth", "checkpoint.pth"],
            optimizer_state_dict=self.optimizer.state_dict(),
        )

        ckp_operator.load_checkpoint(checkpoint_name="model.pth")

    def test_clients_checkpoint_saving(self):
        """Test operations for clients' checkpoint saving."""
        clients_id = [1, 4, 6, 12, 5]

        for save_id in clients_id:
            saved_filename = checkpoint_operator.save_client_checkpoint(
                client_id=save_id,
                model_name="global_model",
                checkpoints_dir="checkpoints/",
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
            )
            logging.info("%s has been saved.", saved_filename)

            loaded_filename, _ = checkpoint_operator.load_client_checkpoint(
                save_id, model_name="global_model", checkpoints_dir="checkpoints/"
            )
            logging.info("%s has been loaded.", loaded_filename)


if __name__ == "__main__":
    unittest.main()
