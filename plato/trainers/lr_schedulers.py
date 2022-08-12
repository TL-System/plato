import bisect
import sys
from collections import OrderedDict

import numpy as np
from torch import optim
from torch import nn
import torch_optimizer as torch_optim

from plato.config import Config
from plato.utils.step import Step


def get_lr_schedule(
    optimizer: optim.Optimizer, iterations_per_epoch: int, train_loader=None
):
    """Returns a learning rate scheduler according to the configuration."""

    registered_schedulers = {
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
        "LambdaLR": optim.lr_scheduler.LambdaLR,
        "MultiStepLR": optim.lr_scheduler.MultiStepLR,
        "StepLR": optim.lr_scheduler.StepLR,
        "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
        "ConstantLR": optim.lr_scheduler.ConstantLR,
        "LinearLR": optim.lr_scheduler.LinearLR,
        "ExponentialLR": optim.lr_scheduler.ExponentialLR,
        "CyclicLR": optim.lr_scheduler.CyclicLR,
        "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
    }

    lr_schedule = Config().trainer.lr_schedule
    lr_params = Config().parameters.learning_rate._asdict()

    # The list containing the learning rate schedule that must be returned or
    # the learning rate schedules that ChainedScheduler or SequentialLR will
    # take as an argument.
    returned_schedules = []

    use_chained = False
    use_sequential = False
    if "ChainedScheduler" in lr_schedule:
        use_chained = True
        lr_schedule = [
            schedule
            for schedule in Config().trainer.lr_schedule.split(",")
            if schedule != ("ChainedScheduler")
        ]
    elif "SequentialLR" in lr_schedule:
        use_sequential = True
        lr_schedule = [
            schedule
            for schedule in Config().trainer.lr_schedule.split(",")
            if schedule != ("SequentialLR")
        ]
    else:
        lr_schedule = [lr_schedule]

    for scheduler in lr_schedule:
        retrived_scheduler = registered_schedulers.get(scheduler, None)
        if retrived_scheduler is None:
            sys.exit("Error: Unknown learning rate scheduler.")

        if scheduler == "CosineAnnealingLR":
            returned_schedules.append(
                retrived_scheduler(
                    optimizer, len(train_loader) * Config().trainer.epochs
                )
            )
        elif scheduler == "LambdaLR":
            lambdas = [lambda it: 1.0]

            if hasattr(Config().parameters.learning_rate, "lr_gamma") and hasattr(
                Config().parameters.learning_rate, "lr_milestone_steps"
            ):
                milestones = [
                    Step.from_str(x, iterations_per_epoch).iteration
                    for x in Config().trainer.lr_milestone_steps.split(",")
                ]
                lambdas.append(
                    lambda it, milestones=milestones: Config().parameters.learning_rate.lr_gamma
                    ** bisect.bisect(milestones, it)
                )

            # Add a linear learning rate warmup if specified
            if hasattr(Config().parameters.learning_rate, "lr_warmup_steps"):
                warmup_iters = Step.from_str(
                    Config().parameters.learning_rate.lr_warmup_steps,
                    iterations_per_epoch,
                ).iteration
                lambdas.append(
                    lambda it, warmup_iters=warmup_iters: min(1.0, it / warmup_iters)
                )
            returned_schedules.append(
                retrived_scheduler(
                    optimizer,
                    lambda it, lambdas=lambdas: np.product([l(it) for l in lambdas]),
                )
            )
        elif scheduler == "MultiStepLR":
            milestones = [
                int(x.split("ep")[0])
                for x in lr_params["lr_milestone_steps"].split(",")
            ]
            returned_schedules.append(
                retrived_scheduler(
                    optimizer, milestones=milestones, gamma=lr_params["gamma"]
                )
            )
        else:
            returned_schedules.append(retrived_scheduler(optimizer, **lr_params))

    if use_chained:
        return optim.lr_scheduler.ChainedScheduler(returned_schedules)
    if use_sequential:
        sequential_milestones = (
            Config().trainer.lr_sequential_milestones
            if hasattr(Config().trainer, "lr_sequential_milestones")
            else 2
        )
        sequential_milestones = [
            int(epoch) for epoch in sequential_milestones.split(",")
        ]

        return optim.lr_scheduler.SequentialLR(
            optimizer, returned_schedules, sequential_milestones
        )
    else:
        return returned_schedules[0]
