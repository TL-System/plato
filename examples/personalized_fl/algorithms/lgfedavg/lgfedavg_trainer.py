"""
A personalized federated learning trainer using LG-FedAvg.

"""


from pflbases import trainer_utils
from plato.trainers import basic
from plato.config import Config


class Trainer(basic.Trainer):
    """A personalized federated learning trainer using the LG-FedAvg algorithm."""

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
        trainer_utils.freeze_model(self.model, Config().algorithm.head_layer_names)
        trainer_utils.activate_model(self.model, Config().algorithm.body_layer_names)
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
            Config().algorithm.body_layer_names,
            log_info=None,
        )
        trainer_utils.activate_model(self.model, Config().algorithm.head_layer_names)

        self.optimizer.step()

        return loss
