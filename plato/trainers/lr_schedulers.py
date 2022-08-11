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

    lambdas = [lambda it: 1.0]

    if hasattr(Config().trainer, "lr_gamma") and hasattr(
        Config().trainer, "lr_milestone_steps"
    ):
        milestones = [
            Step.from_str(x, iterations_per_epoch).iteration
            for x in Config().trainer.lr_milestone_steps.split(",")
        ]
        lambdas.append(
            lambda it, milestones=milestones: Config().trainer.lr_gamma
            ** bisect.bisect(milestones, it)
        )

    # Add a linear learning rate warmup if specified
    if hasattr(Config().trainer, "lr_warmup_steps"):
        warmup_iters = Step.from_str(
            Config().trainer.lr_warmup_steps, iterations_per_epoch
        ).iteration
        lambdas.append(
            lambda it, warmup_iters=warmup_iters: min(1.0, it / warmup_iters)
        )

    registered_schedulers = OrderedDict(
        [
            (
                "CosineAnnealingLR",
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, len(train_loader) * Config().trainer.epochs
                ),
            ),
            (
                "LambdaLR",
                optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda it, lambdas=lambdas: np.product([l(it) for l in lambdas]),
                ),
            ),
            (
                "MultiStepLR",
                optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        int(x.split("ep")[0])
                        for x in Config().trainer.lr_milestone_steps.split(",")
                    ],
                    gamma=Config().trainer.lr_gamma
                    if hasattr(Config().trainer, "gamma")
                    else 0.99,
                ),
            ),
            (
                "StepLR",
                optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=Config().trainer.lr_step_size
                    if hasattr(Config().trainer, "lr_step_size")
                    else 30,
                    gamma=Config().trainer.lr_gamma
                    if hasattr(Config().trainer, "lr_gamma")
                    else 0.1,
                ),
            ),
            (
                "ReduceLROnPlateau",
                optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=Config().trainer.lr_factor
                    if hasattr(Config().trainer, "lr_factor")
                    else 0.1,
                    patience=Config().trainer.lr_patience
                    if hasattr(Config().trainer, "lr_patience")
                    else 10,
                ),
            ),
            (
                "ConstantLR",
                optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=Config().trainer.lr_factor
                    if hasattr(Config().trainer, "lr_factor")
                    else 1.0 / 3.0,
                    total_iters=Config().trainer.lr_total_iters
                    if hasattr(Config().trainer, "lr_total_iters")
                    else 5,
                ),
            ),
            (
                "LinearLR",
                optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=Config().trainer.lr_start_factor
                    if hasattr(Config().trainer, "lr_start_factor")
                    else 1.0 / 3.0,
                    end_factor=Config().trainer.lr_end_factor
                    if hasattr(Config().trainer, "lr_end_factor")
                    else 1.0,
                    total_iters=Config().trainer.lr_total_iters
                    if hasattr(Config().trainer, "lr_total_iters")
                    else 5,
                ),
            ),
            (
                "ExponentialLR",
                optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=Config().trainer.lr_gamma
                ),
            ),
            (
                "CyclicLR",
                optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=Config().trainer.lr_base
                    if hasattr(Config().trainer, "lr_base")
                    else 0.01,
                    max_lr=Config().trainer.lr_max
                    if hasattr(Config().trainer, "lr_max")
                    else 0.1,
                    step_size_up=Config().trainer.lr_step_up
                    if hasattr(Config().trainer, "lr_step_up")
                    else 2000,
                    step_size_down=Config().trainer.lr_step_down
                    if hasattr(Config().trainer, "lr_step_down")
                    else 2000,
                    mode=Config().trainer.mode
                    if hasattr(Config().trainer, "mode")
                    else "triangular",
                ),
            ),
            (
                "CosineAnnealingWarmRestarts",
                optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=Config().trainer.num_iters
                    if hasattr(Config().trainer, "num_iters")
                    else 50,
                ),
            ),
        ]
    )

    lr_schedule = Config().trainer.lr_schedule

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

        returned_schedules.append(retrived_scheduler)

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
