"""
The training and testing loops for PyTorch.
"""
import asyncio
import logging
import multiprocessing as mp
import os
import re
import time

import numpy as np
import torch
from opacus import GradSampleModule
from opacus.privacy_engine import PrivacyEngine
from opacus.validators import ModuleValidator
from plato.config import Config
from plato.models import registry as models_registry
from plato.trainers import base
from plato.utils import optimizers


class Trainer(base.Trainer):
    """A basic federated learning trainer, used by both the client and the server."""

    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__()

        self.training_start_time = time.time()
        self.models_per_epoch = {}

        if model is None:
            model = models_registry.get()

        # Use data parallelism if multiple GPUs are available and the configuration specifies it
        if Config().is_parallel():
            logging.info("Using Data Parallelism.")
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model

        if hasattr(Config().trainer, 'differential_privacy') and Config(
        ).trainer.differential_privacy:
            logging.info("Using differential privacy during training.")

            errors = ModuleValidator.validate(self.model, strict=False)
            if len(errors) > 0:
                self.model = ModuleValidator.fix(self.model)
                errors = ModuleValidator.validate(self.model, strict=False)
                assert len(errors) == 0

            self.model = GradSampleModule(self.model)

    def zeros(self, shape):
        """Returns a PyTorch zero tensor with the given shape."""
        # This should only be called from a server
        assert self.client_id == 0
        return torch.zeros(shape)

    def save_model(self, filename=None, location=None):
        """Saving the model to a file."""
        model_dir = Config(
        ).params['model_dir'] if location is None else location
        model_name = Config().trainer.model_name

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if filename is not None:
            model_path = f'{model_dir}/{filename}'
        else:
            model_path = f'{model_dir}/{model_name}.pth'

        torch.save(self.model.state_dict(), model_path)

        if self.client_id == 0:
            logging.info("[Server #%d] Model saved to %s.", os.getpid(),
                         model_path)
        else:
            logging.info("[Client #%d] Model saved to %s.", self.client_id,
                         model_path)

    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        model_dir = Config(
        ).params['model_dir'] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f'{model_dir}/{filename}'
        else:
            model_path = f'{model_dir}/{model_name}.pth'

        if self.client_id == 0:
            logging.info("[Server #%d] Loading a model from %s.", os.getpid(),
                         model_path)
        else:
            logging.info("[Client #%d] Loading a model from %s.",
                         self.client_id, model_path)

        self.model.load_state_dict(torch.load(model_path))

    def simulate_sleep_time(self):
        """Simulate client's speed by putting it to sleep."""
        sleep_seconds = Config().client_sleep_times[self.client_id - 1]

        # Put this client to sleep
        logging.info("[Client #%d] Going to sleep for %.2f seconds.",
                     self.client_id, sleep_seconds)
        time.sleep(sleep_seconds)
        logging.info("[Client #%d] Woke up.", self.client_id)

    def train_process(self, config, trainset, sampler, cut_layer=None):
        """The main training loop in a federated learning workload, run in
          a separate process with a new CUDA context, so that CUDA memory
          can be released after the training completes.

        Arguments:
        self: the trainer itself.
        config: a dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        """
        tic = time.perf_counter()

        if 'use_wandb' in config:
            import wandb

            run = wandb.init(project="plato",
                             group=str(config['run_id']),
                             reinit=True)

        try:
            custom_train = getattr(self, "train_model", None)

            if callable(custom_train):
                self.train_model(config, trainset, sampler.get(), cut_layer)
            else:
                log_interval = 10
                batch_size = config['batch_size']

                logging.info("[Client #%d] Loading the dataset.",
                             self.client_id)
                _train_loader = getattr(self, "train_loader", None)

                if callable(_train_loader):
                    train_loader = self.train_loader(batch_size, trainset,
                                                     sampler.get(), cut_layer)
                else:
                    train_loader = torch.utils.data.DataLoader(
                        dataset=trainset,
                        shuffle=False,
                        batch_size=batch_size,
                        sampler=sampler.get())

                iterations_per_epoch = np.ceil(len(trainset) /
                                               batch_size).astype(int)
                epochs = config['epochs']

                # Sending the model to the device used for training
                self.model.to(self.device)
                self.model.train()

                # Initializing the loss criterion
                _loss_criterion = getattr(self, "loss_criterion", None)
                if callable(_loss_criterion):
                    loss_criterion = self.loss_criterion(self.model)
                else:
                    loss_criterion = torch.nn.CrossEntropyLoss()

                # Initializing the optimizer
                get_optimizer = getattr(self, "get_optimizer",
                                        optimizers.get_optimizer)
                optimizer = get_optimizer(self.model)

                # Initializing the learning rate schedule, if necessary
                if hasattr(config, 'lr_schedule'):
                    lr_schedule = optimizers.get_lr_schedule(
                        optimizer, iterations_per_epoch, train_loader)
                else:
                    lr_schedule = None

                if 'differential_privacy' in config and config[
                        'differential_privacy']:
                    privacy_engine = PrivacyEngine(accountant='rdp',
                                                   secure_mode=False)

                    self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                        module=self.model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        target_epsilon=config['dp_epsilon']
                        if 'dp_epsilon' in config else 10.0,
                        target_delta=config['dp_delta']
                        if 'dp_delta' in config else 1e-5,
                        epochs=epochs,
                        max_grad_norm=config['dp_max_grad_norm']
                        if 'max_grad_norm' in config else 1.0,
                    )

                for epoch in range(1, epochs + 1):
                    for batch_id, (examples,
                                   labels) in enumerate(train_loader):
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)
                        if 'differential_privacy' in config and config[
                                'differential_privacy']:
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            optimizer.zero_grad()

                        if cut_layer is None:
                            outputs = self.model(examples)
                        else:
                            outputs = self.model.forward_from(
                                examples, cut_layer)

                        loss = loss_criterion(outputs, labels)

                        loss.backward()
                        optimizer.step()

                        if batch_id % log_interval == 0:
                            if self.client_id == 0:
                                logging.info(
                                    "[Server #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(os.getpid(), epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss.data.item()))
                            else:
                                if hasattr(config, 'use_wandb'):
                                    wandb.log({"batch loss": loss.data.item()})

                                logging.info(
                                    "[Client #{}] Epoch: [{}/{}][{}/{}]\tLoss: {:.6f}"
                                    .format(self.client_id, epoch, epochs,
                                            batch_id, len(train_loader),
                                            loss.data.item()))

                    if lr_schedule is not None:
                        lr_schedule.step()

                    if hasattr(optimizer, "params_state_update"):
                        optimizer.params_state_update()

                    # Simulate client's speed
                    if self.client_id != 0 and hasattr(
                            Config().clients, "speed_simulation") and Config(
                            ).clients.speed_simulation:
                        self.simulate_sleep_time()

                    # Saving the model at the end of this epoch to a file so that
                    # it can later be retrieved to respond to server requests
                    # in asynchronous mode when the wall clock time is simulated
                    if hasattr(Config().server, 'request_update') and Config(
                    ).server.request_update:
                        self.model.cpu()
                        training_time = time.perf_counter() - tic
                        filename = f"{self.client_id}_{epoch}_{training_time}.pth"
                        self.save_model(filename)
                        self.model.to(self.device)

        except Exception as training_exception:
            logging.info("Training on client #%d failed.", self.client_id)
            raise training_exception

        if 'max_concurrency' in config:
            self.model.cpu()
            model_type = config['model_name']
            filename = f"{model_type}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

        if 'use_wandb' in config:
            run.finish()

    def train(self, trainset, sampler, cut_layer=None) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.

        Returns:
        float: Elapsed time during training.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        # Set the start time of training in absolute time
        self.training_start_time = time.time()

        if 'max_concurrency' in config:
            self.start_training()
            tic = time.perf_counter()

            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)

            train_proc = mp.Process(target=self.train_process,
                                    args=(config, trainset, sampler,
                                          cut_layer))
            train_proc.start()
            train_proc.join()

            model_name = Config().trainer.model_name
            filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.pth"

            try:
                self.load_model(filename)
            except OSError as error:  # the model file is not found, training failed
                if 'max_concurrency' in config:
                    self.run_sql_statement(
                        "DELETE FROM trainers WHERE run_id = (?)",
                        (self.client_id, ))
                raise ValueError(
                    f"Training on client {self.client_id} failed.") from error

            toc = time.perf_counter()
            self.pause_training()
        else:
            tic = time.perf_counter()
            self.train_process(config, trainset, sampler, cut_layer)
            toc = time.perf_counter()

        training_time = toc - tic

        return training_time

    def test_process(self, config, testset, sampler=None):
        """The testing loop, run in a separate process with a new CUDA context,
        so that CUDA memory can be released after the training completes.

        Arguments:
        config: a dictionary of configuration parameters.
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        """
        self.model.to(self.device)
        self.model.eval()

        # Initialize accuracy to be returned to -1, so that the client can disconnect
        # from the server when testing fails
        accuracy = -1

        try:
            custom_test = getattr(self, "test_model", None)

            if callable(custom_test):
                accuracy = self.test_model(config, testset)
            else:
                if sampler is None:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['batch_size'],
                        shuffle=False)
                # Use a testing set following the same distribution as the training set
                else:
                    test_loader = torch.utils.data.DataLoader(
                        testset,
                        batch_size=config['batch_size'],
                        shuffle=False,
                        sampler=sampler.get())

                correct = 0
                total = 0

                with torch.no_grad():
                    for examples, labels in test_loader:
                        examples, labels = examples.to(self.device), labels.to(
                            self.device)

                        outputs = self.model(examples)

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = correct / total
        except Exception as testing_exception:
            logging.info("Testing on client #%d failed.", self.client_id)
            raise testing_exception

        self.model.cpu()

        if 'max_concurrency' in config:
            model_name = config['model_name']
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.acc"
            self.save_accuracy(accuracy, filename)
        else:
            return accuracy

    def test(self, testset, sampler=None) -> float:
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        if hasattr(Config().trainer, 'max_concurrency'):
            self.start_training()

            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)

            proc = mp.Process(target=self.test_process,
                              args=(
                                  config,
                                  testset,
                                  sampler,
                              ))
            proc.start()
            proc.join()

            accuracy = -1
            try:
                model_name = Config().trainer.model_name
                filename = f"{model_name}_{self.client_id}_{Config().params['run_id']}.acc"
                accuracy = self.load_accuracy(filename)
            except OSError as error:  # the model file is not found, training failed
                raise ValueError(
                    f"Testing on client #{self.client_id} failed.") from error

            self.pause_training()
        else:
            accuracy = self.test_process(config, testset)

        return accuracy

    async def server_test(self, testset, sampler=None):
        """Testing the model on the server using the provided test dataset.

        Arguments:
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        """
        config = Config().trainer._asdict()
        config['run_id'] = Config().params['run_id']

        self.model.to(self.device)
        self.model.eval()

        custom_test = getattr(self, "test_model", None)

        if callable(custom_test):
            return self.test_model(config, testset)

        if sampler is None:
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=config['batch_size'], shuffle=False)
        # Use a testing set following the same distribution as the training set
        else:
            test_loader = torch.utils.data.DataLoader(
                testset,
                batch_size=config['batch_size'],
                shuffle=False,
                sampler=sampler.get())

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                outputs = self.model(examples)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Yield to other tasks in the server
                await asyncio.sleep(0)

        return correct / total

    def obtain_model_update(self, wall_time):
        """
            Obtain a saved model for a particular epoch that finishes just after the provided
            wall clock time is reached.
        """
        # Constructing a list of epochs and training times
        self.models_per_epoch = {}

        for filename in os.listdir(Config().params['model_dir']):
            split = re.match(
                r"(?P<client_id>\d+)_(?P<epoch>\d+)_(?P<training_time>\d+.\d+).pth",
                filename)

            if split is not None:
                epoch = split.group('epoch')
                training_time = split.group('training_time')
                if self.client_id == int(split.group('client_id')):
                    self.models_per_epoch[epoch] = {
                        'training_time': float(training_time),
                        'model_checkpoint': filename
                    }
        # Locate the model at a specific wall clock time
        for epoch in sorted(self.models_per_epoch):
            training_time = self.models_per_epoch[epoch]['training_time']
            model_checkpoint = self.models_per_epoch[epoch]['model_checkpoint']
            if training_time + self.training_start_time > wall_time:
                self.load_model(model_checkpoint)
                logging.info(
                    "[Client #%s] Responding to the server with the model after "
                    "epoch %s finished, at time %s.", self.client_id, epoch,
                    training_time + self.training_start_time)
                return self.model

        return self.model
