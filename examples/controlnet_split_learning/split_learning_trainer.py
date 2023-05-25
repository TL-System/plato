"""Train the ControlNet model with split learning"""
import os
import logging
import torch
from plato.config import Config
from examples.split_learning import split_learning_trainer


class Trainer(split_learning_trainer.Algorithm):
    """The split learning algorithm to train ControlNet."""

    def _client_train_loop(self, examples):
        """Complete the client side training with gradients from server."""
        self.optimizer.zero_grad()
        outputs = self.model.forward(examples)

        # Back propagate with gradients from server
        outputs["control_output"].backward(self.gradients["control"])
        self.optimizer.step()

        # No loss value on the client side
        loss = torch.zeros(1)
        self._loss_tracker.update(loss, examples.size(0))
        return loss

    def _server_train_loop(self, config, examples, labels):
        """The training loop on the server."""
        control = examples["control_output"].detach().requires_grad_(True)

        self.optimizer.zero_grad()
        outputs = self.model(
            control.to(Config().device),
            examples["sd_output"].to(Config().device),
            examples["timestep"],
            examples["cond_txt"].to(Config().device),
        )
        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))
        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()
        self.optimizer.step()

        logging.warning(
            "[Server #%d] Gradients computed with training loss: %.4f",
            os.getpid(),
            loss,
        )
        # Record gradients within the cut layer
        self.cut_layer_grad = [control.grad.clone().detach()]

        return loss

    # test
    # test the validation mse
    # pylint: disable=unused-argument
    def test_model(self, config, testset, sampler=None, **kwargs):
        """
        Evaluates the model with the provided test dataset and test sampler.

        Auguments:
        testset: the test dataset.
        sampler: the test sampler. The default is None.
        kwargs (optional): Additional keyword arguments.
        """
        batch_size = config["batch_size"]

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler
        )

        mses = 0
        total = 0

        self.model.to(self.device)
        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(self.device)

                outputs = self.model(examples)

                outputs = self.process_outputs(outputs)

                mses += torch.nn.functional.mse_loss(outputs, labels).item()

        return mses / total
