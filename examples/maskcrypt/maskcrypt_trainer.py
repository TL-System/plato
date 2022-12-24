"""Customize the default trainer with callbacks to support MaskCrpyt."""
import logging
import os
import torch
import maskcrypt_utils
from plato.config import Config
from plato.callbacks.trainer import TrainerCallback as pTrainerCallback
from transformers import TrainerCallback as hTrainerCallback


class basicTrainerCallback(pTrainerCallback):
    """Trainer callback for the basic trainer in Plato."""

    def on_train_run_end(self, trainer, config):
        """Compute gradients on local data when training is finished."""
        logging.info(
            "[Client #%d] Training completed, computing gradient.", trainer.client_id
        )
        # Set the existing gradients to zeros
        gradients = {}
        [x.grad.zero_() for x in list(trainer.model.parameters())]
        trainer.model.to(trainer.device)
        for idx, (examples, labels) in enumerate(trainer.train_loader):
            examples, labels = examples.to(trainer.device), labels.to(trainer.device)
            outputs = trainer.model(examples)
            loss_criterion = torch.nn.CrossEntropyLoss()
            loss = loss_criterion(outputs, labels)
            loss = loss * (len(labels) / len(trainer.sampler))
            loss.backward()

        param_dict = dict(list(trainer.model.named_parameters()))
        state_dict = trainer.model.state_dict()
        for name in state_dict.keys():
            if name in param_dict:
                gradients[name] = param_dict[name].grad
            else:
                gradients[name] = torch.zeros(state_dict[name].shape)

        maskcrypt_utils.save_gradients(
            gradients=gradients, ppid=os.getppid(), config=Config()
        )


class huggingfaceTrainerCallback(hTrainerCallback):
    """Trainer callback for the huggingface trainer in Plato."""

    def on_train_end(self, args, state, control, **kwargs):
        """Compute gradients on local data when training is finished."""
        logging.info(
            "[Client #%d] Training completed, computing gradient.", trainer.client_id
        )
        model = kwargs["model"]
        train_loader = kwargs["train_dataloader"]

        [x.grad.zero_() for x in list(model.parameters())]
        for _, samples in enumerate(train_loader):
            samples = {k: v.to(model.device) for k, v in samples.items()}
            loss = model(**samples).loss
            loss = loss * (len(samples) / len(train_loader.sampler))
            loss.backward()

        gradients = {}
        param_dict = dict(list(model.named_parameters()))
        state_dict = model.state_dict()
        for name in state_dict.keys():
            if name in param_dict:
                gradients[name] = param_dict[name].grad
            else:
                gradients[name] = torch.zeros(state_dict[name].shape)

        maskcrypt_utils.save_gradients(
            gradients=gradients, config=Config(), ppid=os.getppid()
        )


def get():
    """Return a callback for Plato trainers to compute gradients when training is done."""
    if Config().trainer.type == "HuggingFace":
        return [huggingfaceTrainerCallback]
    else:
        return [basicTrainerCallback]
