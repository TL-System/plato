"""
Customized Trainer for PerFedRLNAS.
"""
import logging
import random
import pickle
import os
import torch
import fednas_specific

from plato.trainers import basic
from plato.config import Config


if Config().trainer.lr_scheduler == "timm":
    BasicTrainer = basic.TrainerWithTimmScheduler
else:
    BasicTrainer = basic.Trainer


class SimuRuntimeError(RuntimeError):
    """Simulated Run time Error"""

class Trainer(BasicTrainer):
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


    def get_loss_criterion(self):
        return fednas_specific.get_nasvit_loss_criterion()

    def get_optimizer(self, model):
        optimizer = fednas_specific.get_optimizer(model)
        return optimizer
    
    def train_run_start(self, config):
        super().train_run_start(config)
        if hasattr(Config().parameters, "simulate"):
            self.sim_mem = (
                random.random() * (self.max_mem - self.min_mem) + self.min_mem
            )
            self.max_mem_allocated = 0

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

    def train_process(self, config, trainset, sampler, **kwargs):
        try:
            self.train_model(config, trainset, sampler.get(), **kwargs)
        except SimuRuntimeError:
            self.exceed_memory = True
            self.utilization = 1
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