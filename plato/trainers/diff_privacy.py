"""
The training and testing loops for PyTorch.
"""
import logging
import time

import torch
from torch.utils.data import Subset

from opacus import GradSampleModule
from opacus.privacy_engine import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

from plato.config import Config
from plato.trainers import basic
from plato.utils import optimizers


class Trainer(basic.Trainer):
    """A differentially private federated learning trainer, used by the client."""

    def __init__(self, model=None):
        """Initializing the trainer with the provided model."""
        super().__init__(model=model)

        self.max_physical_batch_size = (
            Config().trainer.max_physical_batch_size
            if hasattr(Config().trainer, "max_physical_batch_size")
            else 128
        )

        self.make_model_private()

    def make_model_private(self):
        """Make the model private for use with the differential privacy engine."""
        errors = ModuleValidator.validate(self.model, strict=False)
        if len(errors) > 0:
            self.model = ModuleValidator.fix(self.model)
            errors = ModuleValidator.validate(self.model, strict=False)
            assert len(errors) == 0

    def train_model(self, config, trainset, sampler, cut_layer):
        """The default training loop that supports differential privacy."""
        batch_size = config["batch_size"]
        self.sampler = sampler
        tic = time.perf_counter()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        # We have to use poisson sampling to sample the data, rather than the provided sampler.
        # Replacing the poisson sampler with the provided sampler is problematic since it may
        # violate the basic theory of DP-SGD. Therefore, we need to first obtain the train subset
        # based on the provided sampler, and then create a simple dataloader on the train subset
        # without the sampler. We will finally use Opacus to recreate the dataloader from the
        # simple dataloader (with poisson sampling).
        trainset = Subset(trainset, list(sampler))
        self.train_loader = Trainer.get_train_loader(
            batch_size, trainset, sampler=None, cut_layer=cut_layer
        )

        # Initializing the loss criterion
        _loss_criterion = getattr(self, "loss_criterion", None)
        if callable(_loss_criterion):
            loss_criterion = self.loss_criterion(self.model)
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        # Initializing the optimizer
        get_optimizer = getattr(self, "get_optimizer", optimizers.get_optimizer)
        optimizer = get_optimizer(self.model)

        # Initializing the learning rate schedule, if necessary
        if "lr_schedule" in config:
            lr_schedule = optimizers.get_lr_schedule(
                optimizer, len(self.train_loader), self.train_loader
            )
        else:
            lr_schedule = None

        self.model.to(self.device)
        total_epochs = config["epochs"]

        logging.info(
            "[Client #%s] Using differential privacy during training.", self.client_id
        )

        privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)

        self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            target_epsilon=config["dp_epsilon"] if "dp_epsilon" in config else 10.0,
            target_delta=config["dp_delta"] if "dp_delta" in config else 1e-5,
            epochs=total_epochs,
            max_grad_norm=config["dp_max_grad_norm"]
            if "max_grad_norm" in config
            else 1.0,
        )

        self.model.train()

        for self.current_epoch in range(1, total_epochs + 1):
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=self.max_physical_batch_size,
                optimizer=optimizer,
            ) as memory_safe_train_loader:
                self.train_epoch_start(config)
                self.callback_handler.call_event("on_train_epoch_start", self, config)

                for batch_id, (examples, labels) in enumerate(memory_safe_train_loader):
                    examples, labels = examples.to(self.device), labels.to(self.device)
                    optimizer.zero_grad(set_to_none=True)

                    if cut_layer is None:
                        outputs = self.model(examples)
                    else:
                        outputs = self.model.forward_from(examples, cut_layer)

                    loss = loss_criterion(outputs, labels)

                    if "create_graph" in config:
                        loss.backward(create_graph=config["create_graph"])
                    else:
                        loss.backward()

                    optimizer.step()

                    self.train_step_end(config, batch=batch_id, loss=loss)
                    self.callback_handler.call_event(
                        "on_train_step_end", self, config, batch=batch_id, loss=loss
                    )

            if lr_schedule is not None:
                lr_schedule.step()

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

            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

    def train_run_start(self, config):
        """
        Method called at the start of training run.
        """
        self.model = GradSampleModule(self.model)

    def train_run_end(self, config):
        """
        Method called at the end of a training run.
        """
        # After GradSampleModule() conversion, the state_dict names have a `_module` prefix
        # We will need to save the weights with the original layer names without the prefix
        self.model_state_dict = {
            k[8:] if "_module." in k else k: v
            for k, v in self.model.state_dict().items()
        }
