"""
Customized Trainer for PerFedRLNAS.
"""
import logging
import random
import os
import pickle
import re
import torch

import fednas_specific
import fedtools
from model.mobilenetv3_supernet import NasDynamicModel

from plato.trainers import basic
from plato.config import Config


if Config().trainer.lr_scheduler == "timm":
    BasicTrainer = basic.TrainerWithTimmScheduler
else:
    BasicTrainer = basic.Trainer


class SimuRuntimeError(RuntimeError):
    """Simulated Run time Error"""


class TrainerSync(BasicTrainer):
    """Use special optimizer and loss criterion specific for NASVIT."""

    def get_loss_criterion(self):
        return fednas_specific.get_nasvit_loss_criterion()

    def get_optimizer(self, model):
        optimizer = fednas_specific.get_optimizer(model)
        return optimizer


# pylint:disable=too-many-instance-attributes
class TrainerAsync(BasicTrainer):
    """Use special optimizer and loss criterion."""

    def __init__(self, model=None, callbacks=None) -> None:
        super().__init__(model, callbacks)
        self.utilization = 0
        self.exceed_memory = False
        self.sim_mem = None
        if hasattr(Config().parameters, "simulate"):
            self.max_mem = Config().parameters.simulate.max_mem
            self.min_mem = Config().parameters.simulate.min_mem
        else:
            self.max_mem = None
            self.min_mem = None
        self.max_mem_allocated = 0
        config = Config().trainer._asdict()
        self.batch_size = config["batch_size"]
        self.unavailable_batch = 1024
        self.current_config = None

    def get_loss_criterion(self):
        return fednas_specific.get_nasvit_loss_criterion()

    def train_run_start(self, config):
        super().train_run_start(config)
        if hasattr(Config().parameters, "simulate"):
            self.sim_mem = (
                random.random() * (self.max_mem - self.min_mem) + self.min_mem
            )
            self.max_mem_allocated = 0
        self.unavailable_batch = 1024
        self.batch_size = config["batch_size"]

    def perform_forward_and_backward_passes(self, config, examples, labels):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        loss = super().perform_forward_and_backward_passes(config, examples, labels)
        torch.cuda.synchronize(self.device)
        max_mem = torch.cuda.max_memory_allocated(self.device) / 1024**3
        self.max_mem_allocated = max(max_mem, self.max_mem_allocated)
        return loss

    def train_step_end(self, config, batch=None, loss=None):
        super().train_step_end(config, batch, loss)
        if self.max_mem_allocated > self.sim_mem:
            raise SimuRuntimeError
        if self.max_mem_allocated < Config().trainer.mem_usage * self.sim_mem:
            if self.batch_size * 2 <= self.unavailable_batch:
                self.batch_size *= 2

    def adjust_batch_size(self):
        "Decrease the batch size if cannot run."
        self.unavailable_batch = min(self.unavailable_batch, self.batch_size)
        self.batch_size = max(self.batch_size // 2, 1)

    def train_process(self, config, trainset, sampler, **kwargs):
        while True:
            try:
                self.train_model(config, trainset, sampler.get(), **kwargs)
                break
            except SimuRuntimeError:
                self.adjust_batch_size()
            except Exception as training_exception:
                logging.info("Training on client #%d failed.", self.client_id)
                raise training_exception
        if "max_concurrency" in config:
            self.model.cpu()
            model_name = config["model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.pth"
            self.save_model(filename)

        if "max_concurrency" in config:
            model_name = config["model_name"]
            filename = f"{model_name}_{self.client_id}_{config['run_id']}.mem"
            self.save_memory(
                (self.max_mem_allocated, self.exceed_memory, self.sim_mem), filename
            )

    # pylint: disable=unused-argument
    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        return super().get_train_loader(self.batch_size, trainset, sampler)

    @staticmethod
    def save_memory(memory, filename=None):
        """Saving the test memory to a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if filename is not None:
            memory_path = f"{model_path}/{filename}"
        else:
            memory_path = f"{model_path}/{model_name}.mem"

        with open(memory_path, "wb") as file:
            pickle.dump(memory, file)

    @staticmethod
    def load_memory(filename=None):
        """Loading the test memory from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if filename is not None:
            memory_path = f"{model_path}/{filename}"
        else:
            memory_path = f"{model_path}/{model_name}.mem"

        with open(memory_path, "rb") as file:
            memory = pickle.load(file)

        return memory

    def obtain_model_update(self, client_id, requested_time):
        """
        Obtain a saved model for a particular epoch that finishes just after the provided
        wall clock time is reached.
        """
        # Constructing a list of epochs and training times
        models_per_epoch = {}

        for filename in os.listdir(Config().params["model_path"]):
            split = re.match(
                r"(?P<client_id>\d+)_(?P<epoch>\d+)_(?P<training_time>\d+.\d+).pth$",
                filename,
            )

            if split is not None:
                epoch = split.group("epoch")
                training_time = split.group("training_time")
                if client_id == int(split.group("client_id")):
                    models_per_epoch[epoch] = {
                        "training_time": float(training_time),
                        "model_checkpoint": filename,
                    }
        with open(
            f"{Config().params['model_path']}/{client_id}.pkl", "rb"
        ) as history_file:
            subnet_config = pickle.load(history_file)
        # Locate the model at a specific wall clock time
        for epoch in sorted(models_per_epoch, reverse=True):
            model_training_time = models_per_epoch[epoch]["training_time"]
            model_checkpoint = models_per_epoch[epoch]["model_checkpoint"]

            if model_training_time < requested_time:
                model_path = f"{Config().params['model_path']}/{model_checkpoint}"

                pretrained = None
                if torch.cuda.is_available():
                    pretrained = torch.load(model_path)
                else:
                    pretrained = torch.load(
                        model_path, map_location=torch.device("cpu")
                    )
                model = fedtools.sample_subnet_w_config(
                    NasDynamicModel(), subnet_config, False
                )
                model.load_state_dict(pretrained, strict=True)

                logging.info(
                    "[Client #%s] Responding to the server with the model after "
                    "epoch %s finished, at time %s.",
                    client_id,
                    epoch,
                    model_training_time,
                )

                return model

        model_path = f"{Config().params['model_path']}/{model_checkpoint}"

        pretrained = None
        if torch.cuda.is_available():
            pretrained = torch.load(model_path)
        else:
            pretrained = torch.load(model_path, map_location=torch.device("cpu"))
        model = fedtools.sample_subnet_w_config(NasDynamicModel(), subnet_config, False)
        model.load_state_dict(pretrained, strict=True)

        logging.info(
            "[Client #%s] Responding to the server with the model after "
            "epoch %s finished, at time %s.",
            client_id,
            epoch,
            model_training_time,
        )

        return model

    def train_epoch_end(self, config):
        """Method called at the end of a training epoch."""
        super().train_epoch_end(config)
        if (
            hasattr(Config().server, "request_update")
            and Config().server.request_update
        ):
            filename = f"{self.client_id}.pkl"
            model_path = Config().params["model_path"]
            model_path = f"{model_path}/{filename}"
            with open(model_path, "wb") as history_file:
                pickle.dump(self.current_config, history_file)


if hasattr(Config().server, "synchronous") and not Config().server.synchronous:
    Trainer = TrainerAsync
else:
    Trainer = TrainerSync
