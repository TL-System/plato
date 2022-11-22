"""
Returns a learning rate scheduler according to the configuration.
"""
import bisect
import sys
from types import SimpleNamespace
from typing import Union

import numpy as np
from timm import scheduler
from torch import optim

from plato.config import Config


def get(
    optimizer: optim.Optimizer, iterations_per_epoch: int, **kwargs: Union[str, dict]
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

    registered_factories = {
        "timm": scheduler.create_scheduler,
    }

    _scheduler = (
        kwargs["lr_scheduler"]
        if "lr_scheduler" in kwargs
        else Config().trainer.lr_scheduler
    )
    lr_params = (
        kwargs["lr_params"]
        if "lr_params" in kwargs
        else Config().parameters.learning_rate._asdict()
    )

    # First, look up the registered factories of LR schedulers
    if _scheduler in registered_factories:
        scheduler_args = SimpleNamespace(**lr_params)
        scheduler_args.epochs = Config().trainer.epochs
        lr_scheduler, __ = registered_factories[_scheduler](
            args=scheduler_args, optimizer=optimizer
        )
        return lr_scheduler

    # The list containing the learning rate schedulers that must be returned or
    # the learning rate schedulers that ChainedScheduler or SequentialLR will
    # take as an argument.
    returned_schedulers = []

    use_chained = False
    use_sequential = False
    if "ChainedScheduler" in _scheduler:
        use_chained = True
        lr_scheduler = [
            sched for sched in _scheduler.split(",") if sched != ("ChainedScheduler")
        ]
    elif "SequentialLR" in _scheduler:
        use_sequential = True
        lr_scheduler = [
            sched for sched in _scheduler.split(",") if sched != ("SequentialLR")
        ]
    else:
        lr_scheduler = [_scheduler]

    for _scheduler in lr_scheduler:
        retrieved_scheduler = registered_schedulers.get(_scheduler)

        if retrieved_scheduler is None:
            sys.exit("Error: Unknown learning rate scheduler.")

        if _scheduler == "CosineAnnealingLR":
            returned_schedulers.append(
                retrieved_scheduler(
                    optimizer, iterations_per_epoch * Config().trainer.epochs
                )
            )
        elif _scheduler == "LambdaLR":
            lambdas = [lambda it: 1.0]

            if "gamma" in lr_params and "milestone_steps" in lr_params:
                milestones = [
                    Step.from_str(x, iterations_per_epoch).iteration
                    for x in lr_params["milestone_steps"].split(",")
                ]
                lambdas.append(
                    lambda it, milestones=milestones: lr_params["gamma"]
                    ** bisect.bisect(milestones, it)
                )

            # Add a linear learning rate warmup if specified
            if "warmup_steps" in lr_params:
                warmup_iters = Step.from_str(
                    lr_params["warmup_steps"], iterations_per_epoch
                ).iteration
                lambdas.append(
                    lambda it, warmup_iters=warmup_iters: min(1.0, it / warmup_iters)
                )
            returned_schedulers.append(
                retrieved_scheduler(
                    optimizer,
                    lambda it, lambdas=lambdas: np.product([l(it) for l in lambdas]),
                )
            )
        elif _scheduler == "MultiStepLR":
            milestones = [
                int(x.split("ep")[0]) for x in lr_params["milestone_steps"].split(",")
            ]
            returned_schedulers.append(
                retrieved_scheduler(
                    optimizer, milestones=milestones, gamma=lr_params["gamma"]
                )
            )
        else:
            returned_schedulers.append(retrieved_scheduler(optimizer, **lr_params))

    if use_chained:
        return optim.lr_scheduler.ChainedScheduler(returned_schedulers)

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
            optimizer, returned_schedulers, sequential_milestones
        )
    else:
        return returned_schedulers[0]


class Step:
    """Represents a particular step of training."""

    def __init__(self, iteration: int, iterations_per_epoch: int) -> None:
        if iteration < 0:
            raise ValueError("iteration must >= 0.")
        if iterations_per_epoch <= 0:
            raise ValueError("iterations_per_epoch must be > 0.")
        self._iteration = iteration
        self.iterations_per_epoch = iterations_per_epoch

    @staticmethod
    def str_is_zero(s: str) -> bool:
        return s in ["0ep", "0it", "0ep0it"]

    @staticmethod
    def from_iteration(iteration: int, iterations_per_epoch: int) -> "Step":
        return Step(iteration, iterations_per_epoch)

    @staticmethod
    def from_epoch(epoch: int, iteration: int, iterations_per_epoch: int) -> "Step":
        return Step(epoch * iterations_per_epoch + iteration, iterations_per_epoch)

    @staticmethod
    def from_str(s: str, iterations_per_epoch: int) -> "Step":
        """Creates a step from a string that describes the number of epochs, iterations, or both.

        Epochs: '120ep'
        Iterations: '2000it'
        Both: '120ep50it'"""

        if "ep" in s and "it" in s:
            ep = int(s.split("ep")[0])
            it = int(s.split("ep")[1].split("it")[0])
            if s != "{}ep{}it".format(ep, it):
                raise ValueError(f"Malformed string step: {s}")
            return Step.from_epoch(ep, it, iterations_per_epoch)
        elif "ep" in s:
            ep = int(s.split("ep")[0])
            if s != "{}ep".format(ep):
                raise ValueError(f"Malformed string step: {s}")
            return Step.from_epoch(ep, 0, iterations_per_epoch)
        elif "it" in s:
            it = int(s.split("it")[0])
            if s != "{}it".format(it):
                raise ValueError(f"Malformed string step: {s}")
            return Step.from_iteration(it, iterations_per_epoch)
        else:
            raise ValueError(f"Malformed string step: {s}")

    @staticmethod
    def zero(iterations_per_epoch: int) -> "Step":
        return Step(0, iterations_per_epoch)

    @property
    def iteration(self):
        """The overall number of steps of training completed so far."""
        return self._iteration

    @property
    def ep(self):
        """The current epoch of training."""
        return self._iteration // self.iterations_per_epoch

    @property
    def it(self):
        """The iteration within the current epoch of training."""
        return self._iteration % self.iterations_per_epoch

    def _check(self, other):
        if not isinstance(other, Step):
            raise ValueError(f"Invalid type for other: {other}.")
        if self.iterations_per_epoch != other.iterations_per_epoch:
            raise ValueError(
                "Cannot compare steps when epochs are of different lengths."
            )

    def __lt__(self, other):
        self._check(other)
        return self._iteration < other.iteration

    def __le__(self, other):
        self._check(other)
        return self._iteration <= other.iteration

    def __eq__(self, other):
        self._check(other)
        return self._iteration == other.iteration

    def __ne__(self, other):
        self._check(other)
        return self._iteration != other.iteration

    def __gt__(self, other):
        self._check(other)
        return self._iteration > other.iteration

    def __ge__(self, other):
        self._check(other)
        return self._iteration >= other.iteration

    def __str__(self):
        return f"(Iteration {self._iteration}; Iterations per Epoch: {self.iterations_per_epoch})"
