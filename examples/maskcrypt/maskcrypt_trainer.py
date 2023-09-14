"""The customized trainer designed for MaskCrpyt."""
import logging
import os
import torch

from typing import OrderedDict
from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):
    """The trainer with gradient computation when local training is finished."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        self.gradient = OrderedDict()

    def train_run_end(self, config):
        """Compute gradients on local data when training is finished."""
        logging.info(
            "[Client #%d] Training completed, computing gradient.", self.client_id
        )
        # Set the existing gradients to zeros
        [x.grad.zero_() for x in list(self.model.parameters())]
        self.model.to(self.device)
        for idx, (examples, labels) in enumerate(self.train_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            outputs = self.model(examples)
            loss_criterion = torch.nn.CrossEntropyLoss()
            loss = loss_criterion(outputs, labels)
            loss = loss * (len(labels) / len(self.sampler))
            loss.backward()

        param_dict = dict(list(self.model.named_parameters()))
        state_dict = self.model.state_dict()
        for name in state_dict.keys():
            if name in param_dict:
                self.gradient[name] = param_dict[name].grad
            else:
                self.gradient[name] = torch.zeros(state_dict[name].shape)

        model_type = config["model_name"]
        filename = f"{model_type}_gradient_{self.client_id}_{config['run_id']}.pth"
        self._save_gradient(filename)

    def get_gradient(self):
        """Read gradients from file and return to client."""
        model_type = Config().trainer.model_name
        run_id = Config().params["run_id"]
        filename = f"{model_type}_gradient_{self.client_id}_{run_id}.pth"
        return self._load_gradient(filename)

    def _save_gradient(self, filename=None, location=None):
        """Saving the gradients to a file."""
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

        torch.save(self.gradient, model_path)

    def _load_gradient(self, filename=None, location=None):
        """Load gradients from a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.pth"

        return torch.load(model_path)
