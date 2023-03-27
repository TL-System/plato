"""Unit tests for the personalization."""
import os
import unittest

os.environ["config_file"] = "tests/TestsConfig/personalized_config.yml"

from plato.models import registry as models_registry
from plato.config import Config
from plato.trainers import loss_criterion, lr_schedulers, optimizers

from plato.trainers import basic
from plato.algorithms import fedavg
from plato.clients import simple
from plato.servers import fedavg_personalized


class PersonalizationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        __ = Config()

        self.model = models_registry.get()
        self.optimizer = optimizers.get(self.model)
        self.lrs = lr_schedulers.get(self.optimizer, 50)

        self.loss = loss_criterion.get()

    def test_personalized_config(self):
        # define the terms for personalization
        # 1. personalized model
        personalized_model = models_registry.get(
            model_name=Config().trainer.personalized_model_name,
            model_params=Config().parameters.personalized_model._asdict(),
        )
        # 2. optimizer
        personalized_optimizer = optimizers.get(
            personalized_model,
            optimizer_name=Config().trainer.personalized_optimizer,
            optim_params=Config().parameters.personalized_optimizer._asdict(),
        )
        # 3. lr scheduler
        personalized_lrs = lr_schedulers.get(
            personalized_optimizer,
            10,
            lr_scheduler=Config().trainer.personalized_lr_scheduler,
            lr_params=Config().parameters.personalized_learning_rate._asdict(),
        )
        # 4. loss function
        personalized_loss = loss_criterion.get(
            loss_criterion=Config().trainer.personalized_loss_criterion,
            loss_criterion_params=Config().parameters.personalized_loss_criterion._asdict(),
        )

        # test whether loading different hyper-parameters:
        # 1. for the model
        self.assertNotEqual(self.model.__str__(), personalized_model.__str__())
        self.assertNotEqual(
            self.model.fc5.out_features, personalized_model.linear.out_features
        )
        # 2. for the optimizer.
        self.assertNotEqual(
            self.optimizer.param_groups[0]["lr"],
            personalized_optimizer.param_groups[0]["lr"],
        )
        self.assertNotEqual(
            self.optimizer.param_groups[0]["weight_decay"],
            personalized_optimizer.param_groups[0]["weight_decay"],
        )

        # 3. for the lr scheduler.
        self.assertNotEqual(self.lrs.get_last_lr(), personalized_lrs.get_last_lr())

        # 4. for the loss function.
        self.assertNotEqual(self.loss.__str__(), personalized_loss.__str__())

    def test_personalization_running(self):
        """Test whether the personalization runs correctly."""

        trainer = basic.Trainer
        algorithm = fedavg.Algorithm
        client = simple.Client(algorithm=algorithm, trainer=trainer)
        server = fedavg_personalized.Server(algorithm=algorithm, trainer=trainer)

        server.run(client)


if __name__ == "__main__":
    unittest.main()
