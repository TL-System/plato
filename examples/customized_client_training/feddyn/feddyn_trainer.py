"""
An implementation of the FedDyn algorithm.

D. Acar, et al., "Federated Learning Based on Dynamic Regularization," in the
Proceedings of ICLR 2021.

Paper: https://openreview.net/forum?id=B7v4QMR6Z9w

Source code: https://github.com/alpemreacar/FedDyn
"""

import copy
from collections.abc import Sized

import torch

from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.algorithms import (
    FedDynLossStrategyFromConfig,
    FedDynUpdateStrategy,
)
from plato.trainers.strategies.algorithms.feddyn_strategy import (
    positive_integer,
    settings_from_config,
)
from plato.trainers.strategies.data_loader import DefaultDataLoaderStrategy
from plato.trainers.strategies.lr_scheduler import NoSchedulerStrategy
from plato.trainers.strategies.optimizer import DefaultOptimizerStrategy
from plato.trainers.strategies.training_step import (
    DefaultTrainingStepStrategy,
    GradientAccumulationStepStrategy,
)


def realized_partition(sampler):
    """Resolve this attempt's sampler once, without drawing a new partition."""
    if sampler is None:
        raise ValueError("FedDyn requires an explicit realized partition sampler.")
    if isinstance(sampler, torch.utils.data.Sampler):
        realized = sampler
    elif isinstance(sampler, (list, range)):
        realized = torch.utils.data.SubsetRandomSampler(sampler)
    elif hasattr(sampler, "get"):
        realized = sampler.get()
    else:
        realized = sampler
    if not isinstance(realized, Sized):
        raise ValueError("FedDyn requires a finite realized sampler cardinality.")
    count = positive_integer(len(realized), "realized sampler count")
    if hasattr(sampler, "num_samples"):
        declared = positive_integer(sampler.num_samples(), "declared sample count")
        if declared != count:
            raise ValueError("FedDyn declared count differs from realized sampler.")
    return realized, count


class StablePartitionLoader:
    """Check fixed cardinality before fetching each physical batch."""

    def __init__(self, loader, count):
        self.loader, self.count = loader, count

    def __getattr__(self, name):
        loader = self.__dict__.get("loader")
        if loader is None:
            raise AttributeError(name)
        return getattr(loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        while True:
            if len(self.loader.sampler) != self.count:
                raise ValueError(
                    "FedDyn sampler cardinality changed before batch fetch."
                )
            try:
                batch = next(iterator)
            except StopIteration:
                return
            yield batch


class CountedLoader(DefaultDataLoaderStrategy):
    """Bind the realized sampled partition before creating an optimizer."""

    def create_train_loader(self, trainset, sampler, batch_size, context):
        realized, count = realized_partition(sampler)
        loader = super().create_train_loader(trainset, realized, batch_size, context)
        if not isinstance(loader.sampler, Sized):
            raise ValueError(
                "FedDyn requires a finite realized sampler cardinality."
            )
        if (
            len(loader.sampler) != count
            or count != context.state.get("feddyn_attempt_count")
        ):
            raise ValueError("FedDyn loader differs from current attempt partition.")
        if loader.drop_last or len(loader) == 0:
            raise ValueError("FedDyn requires a nonempty loader with drop_last=false.")
        expected = context.state["feddyn_dispatch"]["metadata"][
            "expected_count_or_null"
        ]
        if expected is not None and count != expected:
            raise ValueError("FedDyn realized count differs from dispatched count.")
        context.state["feddyn_num_samples"] = count
        return StablePartitionLoader(loader, count)


class PlainSGD(DefaultOptimizerStrategy):
    """Own only Q, preserving the model's existing parameter objects."""

    def create_optimizer(self, model, context):
        settings = settings_from_config()
        return torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=settings["lr"]
        )


class Trainer(ComposableTrainer):
    """Dedicated dispatch-aware SGD trainer with bounded parent rollback.

    The objective is F+alpha_i<h,w>+alpha_i/2||w-x||². Histories are provisional
    until the server accepts the whole participating set. No client history
    file is a live authority, including successful spawned workers.
    """

    def __init__(self, model=None, callbacks=None):
        """
        Initialize the FedDyn trainer with composition-based strategies.

        Args:
            model: The neural network model to train
            callbacks: Optional list of callback handlers
        """
        settings = settings_from_config()
        window = settings["accumulation_steps"]
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=FedDynLossStrategyFromConfig(),
            model_update_strategy=FedDynUpdateStrategy(),
            optimizer_strategy=PlainSGD(),
            data_loader_strategy=CountedLoader(),
            lr_scheduler_strategy=NoSchedulerStrategy(),
            training_step_strategy=(
                DefaultTrainingStepStrategy()
                if window == 1
                else GradientAccumulationStepStrategy(window)
            ),
        )

    def _transaction(self, action):
        model = self._require_model()
        before = copy.deepcopy(model.state_dict())
        mode = model.training
        device = next(model.parameters()).device
        gradients = {
            n: None if p.grad is None else p.grad.detach().clone()
            for n, p in model.named_parameters()
        }
        parameter_flags = {n: p.requires_grad for n, p in model.named_parameters()}
        owned_tensors = dict(model.named_parameters()) | dict(model.named_buffers())
        trainer_state = {
            key: getattr(self, key)
            for key in ("optimizer", "train_loader", "trainset", "sampler")
            if hasattr(self, key)
        }
        strategy = self.model_update_strategy
        if not isinstance(strategy, FedDynUpdateStrategy):
            raise ValueError(
                "FedDyn requires its single dispatch-aware update strategy."
            )
        strategy_state = copy.deepcopy(strategy.__dict__)
        context_state = dict(self.context.state)
        for key, value in list(context_state.items()):
            if key.startswith("feddyn_"):
                context_state[key] = copy.deepcopy(value)
        try:
            result = action()
            # Even successful shared lifecycle callbacks cannot publish an
            # inconsistent endpoint/state pair after acceptance.
            if strategy.accepted:
                strategy.load_worker_state(
                    strategy.get_worker_state(self.context), self.context
                )
            return result
        except BaseException:
            model.to(device)
            with torch.no_grad():
                for name, tensor in owned_tensors.items():
                    if (
                        tensor.shape != before[name].shape
                        or tensor.dtype != before[name].dtype
                    ):
                        tensor.data = before[name].to(device).clone()
                    else:
                        tensor.copy_(before[name])
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(parameter_flags[name])
                parameter.grad = gradients[name]
            model.train(mode)
            strategy.__dict__.clear()
            strategy.__dict__.update(strategy_state)
            strategy.result = None
            strategy.accepted = False
            self.context.state.clear()
            self.context.state.update(context_state)
            self.context.state.pop("complete_optimizer_step", None)
            self.context.state.pop("optimizer_step_hooks_handled", None)
            for key in ("optimizer", "train_loader", "trainset", "sampler"):
                if key in trainer_state:
                    setattr(self, key, trainer_state[key])
                else:
                    self.__dict__.pop(key, None)
            raise

    def train(self, trainset, sampler, **kwargs):
        """Contain model loads and strategy handoff until the parent accepts."""
        settings_from_config()

        def attempt():
            realized = self._bind_attempt_partition(sampler)
            result = super(Trainer, self).train(trainset, realized, **kwargs)
            strategy = self.model_update_strategy
            if not isinstance(strategy, FedDynUpdateStrategy) or not strategy.accepted:
                raise ValueError("FedDyn local training produced no accepted result.")
            return result

        return self._transaction(attempt)

    def _bind_attempt_partition(self, sampler):
        realized, count = realized_partition(sampler)
        dispatch = self.context.state.get("feddyn_dispatch")
        if dispatch is None:
            raise ValueError(
                "FedDyn training needs a versioned server dispatch; use the dedicated example or explicitly warm-start a new run."
            )
        expected = dispatch["metadata"]["expected_count_or_null"]
        if expected is not None and count != expected:
            raise ValueError("FedDyn realized count differs from dispatched count.")
        # The parent binds this before spawn and passes the same sampler.
        # Worker state may never overwrite this independently observed count.
        self.context.state["feddyn_attempt_count"] = count
        return realized

    def train_model(self, config, trainset, sampler, **kwargs):
        """Apply the same bounded rollback to direct local training failures."""
        if (
            not isinstance(
                self.training_step_strategy,
                (DefaultTrainingStepStrategy, GradientAccumulationStepStrategy),
            )
            or type(self.lr_scheduler_strategy) is not NoSchedulerStrategy
        ):
            raise ValueError(
                "FedDyn supports ordinary/accumulated SGD without AMP, clipping, scheduler or custom multi-update strategies."
            )

        def attempt():
            realized = self._bind_attempt_partition(sampler)
            return super(Trainer, self).train_model(
                config, trainset, realized, **kwargs
            )

        return self._transaction(attempt)
