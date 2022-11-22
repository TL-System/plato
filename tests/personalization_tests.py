"""Unit tests for the personalization."""
import os
import unittest

os.environ["config_file"] = "tests/TestsConfig/personalized_config.yml"

from plato.models import registry as models_registry
from plato.config import Config
from plato.trainers import loss_criterion, lr_schedulers, optimizers


class LrSchedulerTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        __ = Config()

        self.model = models_registry.get()
        self.optimizer = optimizers.get(self.model)
        self.lrs = lr_schedulers.get(self.optimizer, 50)

        self.loss = loss_criterion.get()

    def test_personalized_config(self):
        # define the terms for personalization
        # 1. optimizer
        personalized_optimizer = optimizers.get(
            self.model,
            optimizer_name=Config().trainer.personalized_optimizer,
            optim_params=Config().parameters.personalized_optimizer._asdict(),
        )
        # 2. lr scheduler
        personalized_lrs = lr_schedulers.get(
            personalized_optimizer,
            10,
            lr_scheduler=Config().trainer.personalized_lr_scheduler,
            lr_params=Config().parameters.personalized_learning_rate._asdict(),
        )
        # 3. loss function
        personalized_loss = loss_criterion.get(
            loss_criterion=Config().trainer.personalized_loss_criterion,
            loss_criterion_params=Config().parameters.personalized_loss_criterion._asdict(),
        )

        # test whether loading different hyper-parameters:
        # 1. for the optimizer.
        self.assertNotEqual(
            self.optimizer.param_groups[0]["lr"],
            personalized_optimizer.param_groups[0]["lr"],
        )
        self.assertNotEqual(
            self.optimizer.param_groups[0]["weight_decay"],
            personalized_optimizer.param_groups[0]["weight_decay"],
        )

        # 2. for the lr scheduler.
        self.assertNotEqual(self.lrs.get_last_lr(), personalized_lrs.get_last_lr())

        # 3. for the loss function.
        self.assertNotEqual(self.loss.__str__(), personalized_loss.__str__())


if __name__ == "__main__":
    unittest.main()
