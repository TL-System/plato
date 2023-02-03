"""
HeteroFL algorithm trainer.
"""

import torch
from plato.trainers.basic import Trainer


class ServerTrainer(Trainer):
    """A federated learning trainer of Hermes, used by the server."""

    def test(self, testset, sampler=None, **kwargs) -> float:
        """Because the global model will need to compute the statistics of the model."""
        self.train(testset, sampler, **kwargs)
        return super().test(testset, sampler, **kwargs)


class ClientTrainer(Trainer):
    """A federated learning trainer of Hermes, used by the server."""

    def perform_forward_and_backward_passes(self, config, examples, labels):
        self.optimizer.zero_grad()

        outputs = self.model(examples)

        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        return loss
