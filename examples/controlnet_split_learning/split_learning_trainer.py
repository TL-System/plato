"""Train the ControlNet model with split learning"""
import os
import time
import logging
import torch
from plato.config import Config
from split_learning import split_learning_trainer


class Trainer(split_learning_trainer.Trainer):
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

        self.model.model = self.model.model.to(self.device)
        cond_txt = examples["cond_txt"].to(self.model.model.device)
        t = examples["timestep"].to(self.model.model.device)
        control = control.to(self.model.model.device)
        sd_output = examples["sd_output"]
        for index, items in enumerate(sd_output):
            sd_output[index] = items.to(self.model.model.device)
        self.optimizer.zero_grad()
        outputs = self.model.model(
            control,
            sd_output,
            cond_txt,
            t,
        )
        loss = self._loss_criterion(outputs, labels)
        self._loss_tracker.update(loss, labels.size(0))
        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()
        print(control, control.grad)
        # Record gradients within the cut layer
        self.cut_layer_grad = [control.grad.clone().detach()]
        self.optimizer.step()

        logging.warning(
            "[Server #%d] Gradients computed with training loss: %.4f",
            os.getpid(),
            loss,
        )

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

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """The default training loop when a custom training loop is not supplied."""
        batch_size = config["batch_size"]
        self.sampler = sampler
        tic = time.perf_counter()

        self.run_history.reset()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        self.train_loader = self.get_train_loader(batch_size, trainset, sampler)

        # Initializing the loss criterion
        self._loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        self.optimizer = self.get_optimizer(self.model)
        self.lr_scheduler = self.get_lr_scheduler(config, self.optimizer)
        self.optimizer = self._adjust_lr(config, self.lr_scheduler, self.optimizer)

        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]

        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            for batch_id, (examples, labels) in enumerate(self.train_loader):
                self.train_step_start(config, batch=batch_id)
                self.callback_handler.call_event(
                    "on_train_step_start", self, config, batch=batch_id
                )

                labels = labels.to(self.device)

                loss = self.perform_forward_and_backward_passes(
                    config, examples, labels
                )

                self.train_step_end(config, batch=batch_id, loss=loss)
                self.callback_handler.call_event(
                    "on_train_step_end", self, config, batch=batch_id, loss=loss
                )

            self.lr_scheduler_step()

            if hasattr(self.optimizer, "params_state_update"):
                self.optimizer.params_state_update()

            # Simulate client's speed
            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if (
                hasattr(Config().server, "request_update")
                and Config().server.request_update
            ):
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{self.current_epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            self.run_history.update_metric("train_loss", self._loss_tracker.average)
            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)
