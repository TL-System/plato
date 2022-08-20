"""
The training and testing loops for PyTorch.
"""

import copy
import logging
import multiprocessing as mp
import os
import pickle
import re
import time

import torch
from plato.callbacks.handler import CallbackHandler
from plato.callbacks.trainer import PrintProgressCallback
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import base, loss_criterion, lr_schedulers, optimizers, tracking


class Trainer(base.Trainer):
    """A basic federated learning trainer, used by both the client and the server."""

    def __init__(self, model=None, callbacks=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__()

        self.training_start_time = time.time()
        self.models_per_epoch = {}
        self.model_state_dict = None
        self.current_round = 0

        # Starting from the default trainer callback class, add all supplied trainer callbacks
        self.callbacks = [PrintProgressCallback]
        if callbacks is not None:
            self.callbacks.extend(callbacks)
        self.callback_handler = CallbackHandler(self.callbacks)

        # The run history of performance metrics
        self.run_history = tracking.RunHistory()
        self._loss_tracker = tracking.LossTracker()

        if model is None:
            self.model = models_registry.get()
        else:
            self.model = model()

        self.train_loader = None
        self.sampler = None
        self.lr_scheduler = None
        self.current_epoch = 0

    def zeros(self, shape):
        """Returns a PyTorch zero tensor with the given shape."""
        # This should only be called from a server
        assert self.client_id == 0
        return torch.zeros(shape)

    def save_model(self, filename=None, location=None):
        """Saving the model to a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.pth"

        if self.model_state_dict is None:
            torch.save(self.model.state_dict(), model_path)
        else:
            torch.save(self.model_state_dict, model_path)

        with open(model_path + ".pkl", "wb") as history_file:
            pickle.dump(self.run_history, history_file)

        if self.client_id == 0:
            logging.info("[Server #%d] Model saved to %s.", os.getpid(), model_path)
        else:
            logging.info("[Client #%d] Model saved to %s.", self.client_id, model_path)

    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.pth"

        if self.client_id == 0:
            logging.info(
                "[Server #%d] Loading a model from %s.", os.getpid(), model_path
            )
        else:
            logging.info(
                "[Client #%d] Loading a model from %s.", self.client_id, model_path
            )

        pretrained = None
        if torch.cuda.is_available():
            pretrained = torch.load(model_path)
        else:
            pretrained = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(pretrained, strict=True)

        with open(model_path + ".pkl", "rb") as history_file:
            self.run_history = pickle.load(history_file)

    def simulate_sleep_time(self):
        """Simulate client's speed by putting it to sleep."""
        if not (
            hasattr(Config().clients, "sleep_simulation")
            and Config().clients.sleep_simulation
        ):
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]

            # Put this client to sleep
            logging.info(
                "[Client #%d] Going to sleep for %.2f seconds.",
                self.client_id,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            logging.info("[Client #%d] Woke up.", self.client_id)

    def train_process(self, config, trainset, sampler, **kwargs):
        """
        The main training loop in a federated learning workload, run in a
        separate process with a new CUDA context, so that CUDA memory can be
        released after the training completes.

        Arguments:
        self: the trainer itself.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: The sampler that extracts a partition for this client.
        kwargs (optional): Additional keyword arguments.
        """
        try:
            self.train_model(config, trainset, sampler.get(), **kwargs)
        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        if "max_concurrency" in config:
            self.model.cpu()
            model_name = config["model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """The default training loop when a custom training loop is not supplied."""
        batch_size = config["batch_size"]
        self.sampler = sampler
        tic = time.perf_counter()

        self.run_history.reset()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        self.train_loader = Trainer.get_train_loader(batch_size, trainset, sampler)

        # Initializing the loss criterion
        _loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        optimizer = self.get_optimizer(self.model)
        self.lr_scheduler = self.get_lr_scheduler(config, optimizer)
        optimizer = self._adjust_lr(config, self.lr_scheduler, optimizer)

        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]

        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            for batch_id, (examples, labels) in enumerate(self.train_loader):
                examples, labels = examples.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(examples)

                loss = _loss_criterion(outputs, labels)
                self._loss_tracker.update(loss, labels.size(0))

                if "create_graph" in config:
                    loss.backward(create_graph=config["create_graph"])
                else:
                    loss.backward()

                optimizer.step()

                self.train_step_end(config, batch=batch_id, loss=loss)
                self.callback_handler.call_event(
                    "on_train_step_end", self, config, batch=batch_id, loss=loss
                )

            self.lr_scheduler_step()

            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

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

    def train(self, trainset, sampler, **kwargs) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        kwargs (optional): Additional keyword arguments.

        Returns:
        float: Elapsed time during training.
        """
        config = Config().trainer._asdict()
        config["run_id"] = Config().params["run_id"]

        # Set the start time of training in absolute time
        self.training_start_time = time.time()

        if "max_concurrency" in config:
            tic = time.perf_counter()

            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)

            train_proc = mp.Process(
                target=self.train_process,
                args=(config, trainset, sampler),
                kwargs=kwargs,
            )
            train_proc.start()
            train_proc.join()

            model_name = Config().trainer.model_name
            filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.pth"

            try:
                self.load_model(filename)
            except OSError as error:  # the model file is not found, training failed
                raise ValueError(
                    f"Training on client {self.client_id} failed."
                ) from error

            toc = time.perf_counter()
            self.pause_training()
        else:
            tic = time.perf_counter()
            self.train_process(config, trainset, sampler, **kwargs)
            toc = time.perf_counter()

        training_time = toc - tic

        return training_time

    def test_process(self, config, testset, sampler=None, **kwargs):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        kwargs (optional): Additional keyword arguments.
        """
        self.model.to(self.device)
        self.model.eval()

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            accuracy = self.test_model(config, testset, sampler, **kwargs)
        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        self.model.cpu()

        if "max_concurrency" in config:
            model_name = config["model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy

    def test(self, testset, sampler=None, **kwargs) -> float:
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        kwargs (optional): Additional keyword arguments.
        """
        config = Config().trainer._asdict()
        config["run_id"] = Config().params["run_id"]

        if hasattr(Config().trainer, "max_concurrency"):
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)

            proc = mp.Process(
                target=self.test_process, args=(config, testset, sampler), kwargs=kwargs
            )
            proc.start()
            proc.join()

            accuracy = -1
            try:
                model_name = Config().trainer.model_name
                filename = (
                    f"{model_name}_{self.client_id}_{Config().params['run_id']}.acc"
                )
                accuracy = self.load_accuracy(filename)
            except OSError as error:  # the model file is not found, training failed
                raise ValueError(
                    f"Testing on client #{self.client_id} failed."
                ) from error

            self.pause_training()
        else:
            accuracy = self.test_process(config, testset, **kwargs)

        return accuracy

    def obtain_model_update(self, wall_time):
        """
        Obtain a saved model for a particular epoch that finishes just after the provided
        wall clock time is reached.
        """
        # Constructing a list of epochs and training times
        self.models_per_epoch = {}

        for filename in os.listdir(Config().params["model_path"]):
            split = re.match(
                r"(?P<client_id>\d+)_(?P<epoch>\d+)_(?P<training_time>\d+.\d+).pth",
                filename,
            )

            if split is not None:
                epoch = split.group("epoch")
                training_time = split.group("training_time")
                if self.client_id == int(split.group("client_id")):
                    self.models_per_epoch[epoch] = {
                        "training_time": float(training_time),
                        "model_checkpoint": filename,
                    }
        # Locate the model at a specific wall clock time
        for epoch in sorted(self.models_per_epoch):
            training_time = self.models_per_epoch[epoch]["training_time"]
            model_checkpoint = self.models_per_epoch[epoch]["model_checkpoint"]
            if training_time + self.training_start_time > wall_time:
                self.load_model(model_checkpoint)
                logging.info(
                    "[Client #%s] Responding to the server with the model after "
                    "epoch %s finished, at time %s.",
                    self.client_id,
                    epoch,
                    training_time + self.training_start_time,
                )
                return self.model

        return self.model

    # pylint: disable=unused-argument
    @staticmethod
    def get_train_loader(batch_size, trainset, sampler, **kwargs):
        """
        Creates an instance of the trainloader.

        :param batch_size: the batch size.
        :param trainset: the training dataset.
        :param sampler: the sampler for the trainloader to use.
        """
        return torch.utils.data.DataLoader(
            dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
        )

    # pylint: disable=unused-argument
    def test_model(self, config, testset, sampler, **kwargs):
        """
        Evaluates the model with the provided test dataset and test sampler.

        :param testset: the test dataset.
        :param sampler: the test sampler.
        :param kwargs (optional): Additional keyword arguments.

        """
        batch_size = config["batch_size"]

        if sampler is None:
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False
            )
        else:
            # Use a testing set following the same distribution as the training set
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, sampler=sampler.get()
            )

        correct = 0
        total = 0

        self.model.to(self.device)
        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(self.device)

                outputs = self.model(examples)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def get_optimizer(self, model):
        """Returns the optimizer."""
        return optimizers.get(model)

    def get_lr_scheduler(self, config, optimizer):
        """Returns the learning rate scheduler, if needed."""
        if "lr_scheduler" not in config:
            return None

        return lr_schedulers.get(optimizer, len(self.train_loader))

    def lr_scheduler_step(self):
        """
        Performs a single scheduler step if ``self.lr_scheduler`` has been assigned.
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _adjust_lr(self, config, lr_scheduler, optimizer) -> torch.optim.Optimizer:
        """Returns an optimizer with an initial learning rate that has been
        adjusted according to the current round, so that learning rate
        schedulers can be effective throughout the communication rounds."""

        if "global_lr_scheduler" in config and config["global_lr_scheduler"]:
            global_lr_scheduler = copy.deepcopy(lr_scheduler)

            for __ in range(self.current_round - 1):
                for __ in range(Config().trainer.epochs):
                    global_lr_scheduler.step()

            initial_lr = global_lr_scheduler.get_last_lr()
            optimizer.param_groups[0]["lr"] = initial_lr[0]

        return optimizer

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        return loss_criterion.get()

    def backward(self, config, loss):
        """Perform the backpropagation pass."""

    def train_run_start(self, config):
        """Method called at the start of training run."""

    def train_run_end(self, config):
        """Method called at the end of a training run."""

    def train_epoch_start(self, config):
        """Method called at the beginning of a training epoch."""

    def train_epoch_end(self, config):
        """Method called at the end of a training epoch."""

    def train_step_end(self, config, batch=None, loss=None):
        """
        Method called at the end of a training step.

        :param batch: the current batch of training data.
        :param loss: the loss computed in the current batch.
        """


class TrainerWithTimmScheduler(Trainer):
    """
    Subclass of the :class:`Trainer` that works with `timm schedulers
    <https://fastai.github.io/timmdocs/schedulers>` instead of standard PyTorch
    learning rate schedulers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_updates = None

    def train_epoch_start(self, config):
        """Method called at the beginning of a training epoch."""
        super().train_epoch_start(config)
        self.num_updates = self.current_epoch * len(self.train_loader)

    def lr_scheduler_step(self):
        self.num_updates += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step_update(num_updates=self.num_updates)

    def train_epoch_end(self, config):
        """Method called at the end of a training epoch."""
        super().train_epoch_end(config)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.current_epoch + 1)
