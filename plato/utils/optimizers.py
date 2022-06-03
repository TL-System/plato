"""
Optimizers for training workloads.
"""

import bisect
import sys

import numpy as np
from torch import optim
from torch import nn
import torch_optimizer as torch_optim

from plato.config import Config
from plato.utils.step import Step

optimizers_pool = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "Adadelta": optim.Adadelta,
    "Adahessian": torch_optim.Adahessian
}

lr_schedulers_pool = {
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "LambdaLR": optim.lr_scheduler.LambdaLR,
    "StepLR": optim.lr_scheduler.StepLR,
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau
}


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.log = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """ Reset the meter collector. """
        self.log.append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num_of_items=1):
        """ Update the meter collector. """
        self.val = val
        self.sum += val * num_of_items
        self.count += num_of_items
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_optimizer(model) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""
    if Config().trainer.optimizer == 'SGD':
        return optim.SGD(model.parameters(),
                         lr=Config().trainer.learning_rate,
                         momentum=Config().trainer.momentum,
                         weight_decay=Config().trainer.weight_decay)
    elif Config().trainer.optimizer == 'Adam':
        return optim.Adam(model.parameters(),
                          lr=Config().trainer.learning_rate,
                          weight_decay=Config().trainer.weight_decay)
    elif Config().trainer.optimizer == 'Adadelta':
        return optim.Adadelta(model.parameters(),
                              lr=Config().trainer.learning_rate,
                              rho=Config().trainer.rho,
                              eps=float(Config().trainer.eps),
                              weight_decay=Config().trainer.weight_decay)
    elif Config().trainer.optimizer == 'AdaHessian':
        return torch_optim.Adahessian(
            model.parameters(),
            lr=Config().trainer.learning_rate,
            betas=(Config().trainer.momentum_b1, Config().trainer.momentum_b2),
            eps=float(Config().trainer.eps),
            weight_decay=Config().trainer.weight_decay,
            hessian_power=Config().trainer.hessian_power,
        )

    raise ValueError(f'No such optimizer: {Config().trainer.optimizer}')


def get_lr_schedule(optimizer: optim.Optimizer,
                    iterations_per_epoch: int,
                    train_loader=None):
    """Returns a learning rate scheduler according to the configuration."""
    if Config().trainer.lr_schedule == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            len(train_loader) * Config().trainer.epochs)
    elif Config().trainer.lr_schedule == 'LambdaLR':
        lambdas = [lambda it: 1.0]

        if hasattr(Config().trainer, 'lr_gamma') and hasattr(
                Config().trainer, 'lr_milestone_steps'):
            milestones = [
                Step.from_str(x, iterations_per_epoch).iteration
                for x in Config().trainer.lr_milestone_steps.split(',')
            ]
            lambdas.append(lambda it: Config().trainer.lr_gamma**bisect.bisect(
                milestones, it))

        # Add a linear learning rate warmup if specified
        if hasattr(Config().trainer, 'lr_warmup_steps'):
            warmup_iters = Step.from_str(Config().trainer.lr_warmup_steps,
                                         iterations_per_epoch).iteration
            lambdas.append(lambda it: min(1.0, it / warmup_iters))

        # Combine the lambdas
        return optim.lr_scheduler.LambdaLR(
            optimizer, lambda it: np.product([l(it) for l in lambdas]))
    elif Config().trainer.lr_schedule == 'StepLR':
        step_size = Config().trainer.lr_step_size if hasattr(
            Config().trainer, 'lr_step_size') else 30
        gamma = Config().trainer.lr_gamma if hasattr(Config().trainer,
                                                     'lr_gamma') else 0.1
        return optim.lr_scheduler.StepLR(optimizer,
                                         step_size=step_size,
                                         gamma=gamma)
    elif Config().trainer.lr_schedule == 'ReduceLROnPlateau':
        factor = Config().trainer.lr_factor if hasattr(Config().trainer,
                                                       'lr_factor') else 0.1
        patience = Config().trainer.lr_patience if hasattr(
            Config().trainer, 'lr_patience') else 10
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode='min',
                                                    factor=factor,
                                                    patience=patience)
    else:
        sys.exit('Error: Unknown learning rate scheduler.')


def insert_parameter(defined_parameters,
                     parameter_prefix,
                     parameter_name,
                     desired_parameter_name=None,
                     is_manority=True,
                     default_value=None):
    """ Insert one parameter to the defined_parameters.

            The defined parameters in <defined_parameters> have the
        high priority. Only when the <parameter_name> is not included
        in the <defined_parameters>, the parameter with same name defined
        in the config file will be assigned to the <defined_parameters>.

            If is_manority is True, the <parameter_name> must be included in
        the <defined_parameters> or the config file. Thus, the corresponding
        value from these three sources can be assigned to the <defined_parameters>.
        If the <default_value> is set, only when the <parameter_name>
        is not included in <defined_parameters> and the config file, the default value
        will be assigned to this parameter.

            If is_manority is False, only if <parameter_name> appears in
        config file, it will be assigned to <defined_parameters>.

            In some cases, it is desired to change the <parameter_name> to
        the desired one <desired_parameter_name>.
    """
    if parameter_prefix is None:
        parameter_config_name = parameter_name
    else:
        parameter_config_name = parameter_prefix + parameter_name
    # insert the parameter if it does not exist in defined_parameters
    # only insert parameter when it is not defined
    if parameter_name not in defined_parameters:
        if hasattr(Config().trainer, parameter_config_name):

            parameter_value = getattr(Config().trainer, parameter_config_name)
            defined_parameters[parameter_name] = parameter_value
        else:
            if is_manority:
                if default_value is None:
                    sys.exit(f"{parameter_name}: Not Found")
                else:
                    defined_parameters[parameter_name] = default_value
    # change the parameter name to desired one if needed
    if desired_parameter_name is not None and parameter_name in defined_parameters:
        if parameter_name != desired_parameter_name:
            defined_parameters[
                desired_parameter_name] = defined_parameters.pop(
                    parameter_name)
    return defined_parameters


def get_dynamic_optimizer(model, **kwargs) -> optim.Optimizer:
    """Obtain the optimizer used for training the model."""

    # obtain the prefix if it is provided
    # 'prefix' is a very important indicator to declare
    # which part of parameters will be used
    #   For example, in the self-supervised learning,
    # we have two optimizers following different parameters in
    # the config file. For the optimizer of evaluation part,
    # its corresponding parameters' name start with "pers_".

    prefix = None
    if "prefix" in kwargs:
        prefix = kwargs.pop("prefix")

    # it is expected to set these basic params in the
    # function's arguments
    # otherwise, the default in the config file will
    # be used.
    kwargs = insert_parameter(kwargs,
                              prefix,
                              "learning_rate",
                              "lr",
                              is_manority=True)
    kwargs = insert_parameter(kwargs, prefix, "weight_decay", is_manority=True)
    kwargs = insert_parameter(kwargs, prefix, "optimizer", is_manority=True)
    optimizer_name = kwargs.pop("optimizer")

    # add different parameters for different optimizers
    if optimizer_name == 'SGD':
        kwargs = insert_parameter(kwargs,
                                  prefix,
                                  "momentum",
                                  is_manority=False)

    if optimizer_name == 'Adam':
        kwargs = insert_parameter(kwargs, prefix, "betas", is_manority=False)
        kwargs = insert_parameter(kwargs, prefix, "eps", is_manority=False)

    if optimizer_name == 'Adadelta':
        kwargs = insert_parameter(kwargs, prefix, "rho", is_manority=False)
        kwargs = insert_parameter(kwargs, prefix, "eps", is_manority=False)

    if optimizer_name == 'AdaHessian':
        kwargs = insert_parameter(kwargs, prefix, "betas", is_manority=False)
        kwargs = insert_parameter(kwargs, prefix, "eps", is_manority=False)
        kwargs = insert_parameter(kwargs,
                                  prefix,
                                  "hessian_power",
                                  is_manority=False)

    supported_optimziers = list(optimizers_pool.keys())
    if optimizer_name not in supported_optimziers:
        raise ValueError(f'No such optimizer: {optimizer_name}')

    return optimizers_pool[optimizer_name](model.parameters(), **kwargs)


def get_dynamic_lr_schedule(optimizer: optim.Optimizer,
                            iterations_per_epoch: int,
                            train_loader=None,
                            **kwargs):
    """Returns a learning rate scheduler according to the configuration."""

    # obtain the prefix if it is provided
    # 'prefix' is a very important indicator to declare
    # which part of parameters will be used
    #   For example, in the self-supervised learning,
    # we have two optimizers following different parameters in
    # the config file. For the optimizer of evaluation part,
    # its corresponding parameters' name start with "pers_".

    prefix = None
    if "prefix" in kwargs:
        prefix = kwargs.pop("prefix")

    kwargs = insert_parameter(kwargs, prefix, "lr_schedule", is_manority=True)
    lr_schedule = kwargs.pop("lr_schedule")

    if lr_schedule == 'CosineAnnealingLR':
        kwargs = insert_parameter(kwargs, prefix, "T_max", is_manority=False)
        kwargs = insert_parameter(kwargs, prefix, "epochs", is_manority=True)

        if "T_max" not in kwargs:
            epochs = kwargs.pop("epochs")
            kwargs["T_max"] = len(train_loader) * epochs

    if lr_schedule == "LambdaLR":
        kwargs = insert_parameter(kwargs,
                                  prefix,
                                  "lr_gamma",
                                  is_manority=False)
        kwargs = insert_parameter(kwargs,
                                  prefix,
                                  "lr_milestone_steps",
                                  is_manority=False)
        kwargs = insert_parameter(kwargs,
                                  prefix,
                                  "lr_warmup_steps",
                                  is_manority=False)
        lambdas = [lambda it: 1.0]
        if "lr_gamma" in kwargs and "lr_milestone_steps" in kwargs:
            lr_gamma = kwargs.pop("lr_gamma")
            lr_milestone_steps = kwargs.pop("lr_milestone_steps")
            milestones = [
                Step.from_str(x, iterations_per_epoch).iteration
                for x in lr_milestone_steps.split(',')
            ]
            lambdas.append(lambda it: lr_gamma**bisect.bisect(milestones, it))
        if "lr_warmup_steps" in kwargs:
            lr_warmup_steps = kwargs.pop("lr_warmup_steps")
            warmup_iters = Step.from_str(lr_warmup_steps,
                                         iterations_per_epoch).iteration
            lambdas.append(lambda it: min(1.0, it / warmup_iters))
        kwargs["lr_lambda"] = lambda it: np.product([l(it) for l in lambdas])

    if lr_schedule == "StepLR":
        kwargs = insert_parameter(kwargs,
                                  prefix,
                                  "lr_step_size",
                                  is_manority=True,
                                  desired_parameter_name="step_size",
                                  default_value=30)
        kwargs = insert_parameter(kwargs,
                                  prefix,
                                  "lr_gamma",
                                  is_manority=True,
                                  desired_parameter_name="gamma",
                                  default_value=0.1)

    if lr_schedule == "ReduceLROnPlateau":
        kwargs = insert_parameter(kwargs,
                                  prefix,
                                  "lr_factor",
                                  is_manority=True,
                                  desired_parameter_name="factor",
                                  default_value=0.1)
        kwargs = insert_parameter(kwargs,
                                  prefix,
                                  "lr_patience",
                                  is_manority=True,
                                  desired_parameter_name="patience",
                                  default_value=10)
        kwargs["mode"] = "min"

    supported_schedulers = list(lr_schedulers_pool.keys())
    if lr_schedule not in supported_schedulers:
        raise ValueError(f'Unknown learning rate lscheduler: {lr_schedule}')

    return lr_schedulers_pool[lr_schedule](optimizer, **kwargs)


def get_loss_criterion():
    """Obtain the loss criterion used for training the model."""
    if Config().trainer.loss_criterion == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()
