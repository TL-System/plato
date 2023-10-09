"""
A personalized federated learning trainer using LG-FedAvg.

"""

from plato.config import Config

from pflbases import personalized_trainer
from pflbases import trainer_utils


class Trainer(personalized_trainer.Trainer):
    """A personalized federated learning trainer using the LG-FedAvg algorithm."""

    def preprocess_personalized_model(self, config):
        """In LG-FedAvg, only during the personalization process,
        the completed model will be assigned to the personalization model
        for direct evaluation."""
        if self.personalized_learning:
            self.copy_model_to_personalized_model(config)

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Performing one iteration of LG-FedAvg."""
        self.optimizer.zero_grad()

        outputs = self.forward_examples(examples)

        loss = self._loss_criterion(outputs, labels)

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        # first freeze the head and optimize the body
        trainer_utils.freeze_model(
            self.model,
            Config().algorithm.head_modules_name,
            log_info=None,
        )
        trainer_utils.activate_model(self.model, Config().algorithm.body_modules_name)
        self.optimizer.step()

        # repeat the same optimization relying the optimized
        # body of the model
        self.optimizer.zero_grad()

        outputs = self.forward_examples(examples)

        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        # first freeze the head and optimize the body
        trainer_utils.freeze_model(
            self.model,
            Config().algorithm.body_modules_name,
            log_info=None,
        )
        trainer_utils.activate_model(self.model, Config().algorithm.head_modules_name)

        self.optimizer.step()

        return loss
