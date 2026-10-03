"""Native clipping, optimizer aliases, model modes and device regressions."""

from collections.abc import Sequence
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from plato.trainers.mlx import (
    ComposableMLXTrainer,
    DefaultMLXLossStrategy,
    DefaultMLXOptimizerStrategy,
    DefaultMLXTrainingStepStrategy,
    MLXLossCriterionStrategy,
    MLXTrainingContext,
    _to_mx_array,
)
from tests.mlx_native.helpers import native_config, no_device_flags


class Vector(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.zeros(2)

    def __call__(self, x):
        return (self.weight * x).sum()


def test_clipping_changes_native_sgd_update():
    model = Vector()
    optimizer = optim.SGD(learning_rate=1.0)
    strategy = DefaultMLXTrainingStepStrategy(clip_grad_norm=2)
    context = MLXTrainingContext(model=model)
    strategy.setup(context)
    strategy.training_step(
        model, optimizer, mx.array([3.0, 4.0]), None, lambda output, _: output, context
    )
    np.testing.assert_allclose(
        np.asarray(model.weight), -np.array([3.0, 4.0]) * (2 / (5 + 1e-6)), rtol=1e-6
    )


@pytest.mark.parametrize("name", ["momentum", "rmsprop"])
def test_native_optimizer_aliases(tmp_path, name):
    with native_config(tmp_path):
        kwargs = {"learning_rate": 0.1}
        if name == "momentum":
            kwargs["momentum"] = 0.9
        strategy = DefaultMLXOptimizerStrategy(name, **kwargs)
        optimizer = strategy.create_optimizer(Vector(), MLXTrainingContext())
        assert isinstance(optimizer, optim.SGD if name == "momentum" else optim.RMSprop)


def test_no_flag_factory_uses_caller_native_device(tmp_path):
    with native_config(tmp_path):
        no_device_flags()
        trainer = ComposableMLXTrainer(model=Vector)
        assert trainer.context.device == mx.default_device()


def test_evaluation_restores_submodule_modes_on_error(tmp_path):
    class Stochastic(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.5)
            self.linear = nn.Linear(2, 2)
            self.observed = []

        def __call__(self, x):
            self.observed.append(self.training)
            raise RuntimeError("evaluation failure")

    with native_config(tmp_path):
        trainer = ComposableMLXTrainer(model=Stochastic)
        model = cast(Stochastic, trainer.model)
        model.dropout.eval()
        with pytest.raises(RuntimeError, match="evaluation failure"):
            trainer.test_model({"batch_size": 1}, [(np.ones(2), 0)])
        assert model.observed == [False]
        assert model.training
        assert not model.dropout.training


@pytest.mark.parametrize(
    "threshold,gradient",
    [
        (None, [3.0, 4.0]),
        (8.0, [3.0, 4.0]),
        (2.0, [0.0, 0.0]),
        (0.0, [3.0, 4.0]),
        (2.0, [3.0, 4.0]),
    ],
    ids=["disabled", "below-threshold", "zero-gradient", "zero-threshold", "clipped"],
)
def test_clipping_native_update_count_and_original_norm(threshold, gradient):
    class CountingSGD(optim.SGD):
        def update(self, model, gradients):
            self.calls = getattr(self, "calls", 0) + 1
            return super().update(model, gradients)

    model = Vector()
    optimizer = CountingSGD(learning_rate=0.5)
    context = MLXTrainingContext(model=model)
    strategy = DefaultMLXTrainingStepStrategy(clip_grad_norm=threshold)
    strategy.setup(context)
    strategy.training_step(
        model, optimizer, mx.array(gradient), None, lambda output, _: output, context
    )
    norm = np.linalg.norm(gradient)
    scale = 1 if threshold is None else min(threshold / (norm + 1e-6), 1)
    np.testing.assert_allclose(
        np.asarray(model.weight), -0.5 * np.array(gradient) * scale, rtol=1e-6
    )
    assert optimizer.calls == 1
    assert int(optimizer.state["step"].item()) == 1
    if threshold is not None:
        assert context.state["grad_norm"] == pytest.approx(norm)


@pytest.mark.parametrize(
    "threshold",
    [-1.0, float("nan"), float("inf"), -float("inf")],
    ids=["negative", "nan", "positive-infinity", "negative-infinity"],
)
def test_invalid_clipping_rejected_before_updates(threshold):
    strategy = DefaultMLXTrainingStepStrategy(clip_grad_norm=threshold)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        strategy.setup(MLXTrainingContext())


@pytest.mark.parametrize(
    "gradient",
    [[float("nan"), 1.0], [float("inf"), 1.0]],
    ids=["nan-gradient", "infinite-gradient"],
)
def test_nonfinite_clipped_gradients_do_not_update(gradient):
    model = Vector()
    optimizer = optim.SGD(learning_rate=0.5)
    strategy = DefaultMLXTrainingStepStrategy(clip_grad_norm=2.0)
    context = MLXTrainingContext(model=model)
    strategy.setup(context)
    with pytest.raises(ValueError, match="finite gradients"):
        strategy.training_step(
            model,
            optimizer,
            mx.array(gradient),
            None,
            lambda output, _: output,
            context,
        )
    np.testing.assert_array_equal(np.asarray(model.weight), np.zeros(2))
    assert int(optimizer.state["step"].item()) == 0


def test_requested_jit_is_not_silently_ignored():
    with pytest.raises(ValueError, match="eager"):
        DefaultMLXTrainingStepStrategy(jit=True).setup(MLXTrainingContext())


def test_custom_training_strategy_retains_clipping_control(tmp_path):
    class LinearLoss(MLXLossCriterionStrategy):
        def compute_loss(self, outputs, labels, context):
            return outputs

    with native_config(tmp_path, clip_grad_norm=2):
        custom = DefaultMLXTrainingStepStrategy(clip_grad_norm=None)
        trainer = ComposableMLXTrainer(
            model=Vector,
            loss_strategy=LinearLoss(),
            training_step_strategy=custom,
            optimizer_strategy=DefaultMLXOptimizerStrategy("sgd", learning_rate=1.0),
        )
        data = [(np.array([3.0, 4.0], dtype=np.float32), 0)]
        trainer.train_model({"epochs": 1, "batch_size": 1}, data, None)
        model = cast(Vector, trainer.model)
        np.testing.assert_array_equal(np.asarray(model.weight), [-3.0, -4.0])
        assert "grad_norm" not in trainer.context.state


@pytest.mark.parametrize(
    "flags",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["normal-metal", "cpu", "mps", "cpu-precedence"],
)
@pytest.mark.parametrize(
    "custom", [False, True], ids=["normal-caller", "cpu-custom-stream"]
)
def test_native_device_stream_scopes_factory_train_eval_load(
    tmp_path, flags, custom, monkeypatch
):
    from plato.config import Config
    from plato.trainers import mlx as runtime

    events = []

    def observe(label):
        events.append(
            (label, mx.default_device(), mx.default_stream(mx.default_device()))
        )

    class Observed(nn.Module):
        def __init__(self):
            super().__init__()
            observe("factory")
            self.linear = nn.Linear(2, 2)

        def __call__(self, x):
            observe("train" if self.training else "eval")
            return self.linear(x)

    def converting(x):
        observe("load")
        return _to_mx_array(x)

    with native_config(tmp_path):
        Config.args.cpu, Config.args.mps = flags
        caller_device = mx.cpu if custom else mx.gpu
        caller_stream = (
            mx.new_stream(caller_device) if custom else mx.default_stream(caller_device)
        )
        with mx.stream(caller_stream):
            before_device = mx.default_device()
            before_streams = (mx.default_stream(mx.cpu), mx.default_stream(mx.gpu))
            target = mx.cpu if flags[0] else mx.gpu if flags[1] else caller_device
            expected_stream = mx.default_stream(target)
            trainer = ComposableMLXTrainer(model=Observed)
            model = cast(Observed, trainer.model)
            data = [
                (np.ones(2, dtype=np.float32), 0),
                (np.zeros(2, dtype=np.float32), 1),
            ]
            model.eval()
            trainer.train_model({"epochs": 1, "batch_size": 2}, data, None)
            trainer.test_model({"batch_size": 2}, data)
            monkeypatch.setattr(runtime, "_to_mx_array", converting)
            trainer._apply_model_state(trainer._capture_model_state())
            assert {label for label, _, _ in events} == {
                "factory",
                "train",
                "eval",
                "load",
            }
            assert all(
                device == target and stream == expected_stream
                for _, device, stream in events
            )
            assert mx.default_device() == before_device
            assert (
                mx.default_stream(mx.cpu),
                mx.default_stream(mx.gpu),
            ) == before_streams

            def fail(_):
                observe("load")
                raise RuntimeError("conversion failure")

            monkeypatch.setattr(runtime, "_to_mx_array", fail)
            with pytest.raises(RuntimeError, match="conversion failure"):
                trainer._apply_model_state(trainer._capture_model_state())
            assert mx.default_device() == before_device
            assert (
                mx.default_stream(mx.cpu),
                mx.default_stream(mx.gpu),
            ) == before_streams
            monkeypatch.setattr(runtime, "_to_mx_array", _to_mx_array)
            loss_strategy = cast(DefaultMLXLossStrategy, trainer.loss_strategy)
            original_loss = loss_strategy.loss_fn

            def fail_loss(*args):
                observe("train")
                raise RuntimeError("training failure")

            loss_strategy.loss_fn = fail_loss
            with pytest.raises(RuntimeError, match="training failure"):
                trainer.train_model({"epochs": 1, "batch_size": 2}, data, None)
            loss_strategy.loss_fn = original_loss
            assert mx.default_device() == before_device
            assert (
                mx.default_stream(mx.cpu),
                mx.default_stream(mx.gpu),
            ) == before_streams


def test_factory_failure_restores_device_streams_and_rngs(tmp_path):
    import random

    import torch

    from plato.config import Config

    with native_config(tmp_path, model_seed=17):
        Config.args.cpu, Config.args.mps = True, False
        stream = mx.new_stream(mx.gpu)
        with mx.stream(stream):
            before_streams = (mx.default_stream(mx.cpu), mx.default_stream(mx.gpu))
            before_random = random.getstate()
            before_numpy = np.random.get_state()
            before_torch = torch.get_rng_state().clone()
            before_mlx = np.array(cast(Sequence[mx.array], mx.random.state)[0], copy=True)

            def fail_factory():
                assert mx.default_device() == mx.cpu
                random.random()
                np.random.uniform()
                torch.rand(1)
                mx.random.normal((1,))
                raise RuntimeError("factory failure")

            with pytest.raises(RuntimeError, match="factory failure"):
                ComposableMLXTrainer(model=fail_factory)
            assert mx.default_device() == mx.gpu
            assert (
                mx.default_stream(mx.cpu),
                mx.default_stream(mx.gpu),
            ) == before_streams
            assert random.getstate() == before_random
            np.testing.assert_array_equal(np.random.get_state()[1], before_numpy[1])
            assert torch.equal(torch.get_rng_state(), before_torch)
            np.testing.assert_array_equal(
                np.asarray(cast(Sequence[mx.array], mx.random.state)[0]), before_mlx
            )


def test_unavailable_explicit_metal_is_clear(tmp_path, monkeypatch):
    from plato.config import Config

    with native_config(tmp_path):
        Config.args.cpu, Config.args.mps = False, True
        monkeypatch.setattr(mx.metal, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="Metal.*unavailable"):
            ComposableMLXTrainer(model=Vector)


def test_repeated_evaluation_is_deterministic_and_preserves_statistics(tmp_path):
    observed = []

    class Stochastic(nn.Module):
        def __init__(self):
            super().__init__()
            self.batchnorm = nn.BatchNorm(2)
            self.dropout = nn.Dropout(0.75)
            self.linear = nn.Linear(2, 2)

        def __call__(self, x):
            output = self.linear(self.dropout(self.batchnorm(x)))
            observed.append(np.array(output, copy=True))
            return output

    with native_config(tmp_path, model_seed=17):
        trainer = ComposableMLXTrainer(model=Stochastic)
        model = cast(Stochastic, trainer.model)
        data = [(np.array([i, i + 1], dtype=np.float32), i % 2) for i in range(8)]
        model.eval()
        trainer.train_model({"epochs": 1, "batch_size": 8}, data, None)
        assert model.training
        stats = np.array(model.batchnorm.running_mean, copy=True)
        model.dropout.eval()
        observed.clear()
        first = trainer.test_model({"batch_size": 8}, data)
        second = trainer.test_model({"batch_size": 8}, data)
        assert first == second
        np.testing.assert_array_equal(observed[0], observed[1])
        np.testing.assert_array_equal(
            np.asarray(model.batchnorm.running_mean), stats
        )
        assert model.training and model.batchnorm.training
        assert not model.dropout.training
