"""Independent rational FedDyn references through the actual client/server path."""

import asyncio
import copy
import importlib.util
import json
import random
import shlex
import subprocess
import sys
import tomllib
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from plato.callbacks.server import ServerCallback
from plato.callbacks.trainer import TrainerCallback
from plato.serialization.safetensor import deserialize_tree, serialize_tree
from plato.trainers.composable import ComposableTrainer
from plato.trainers.strategies.training_step import DefaultTrainingStepStrategy
from tests.integration.utils import configure_environment

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples/customized_client_training/feddyn"
)


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


feddyn_trainer = load_example("feddyn_trainer")
feddyn_client = load_example("feddyn_client")


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.double))

    def forward(self, examples):
        return self.theta.expand_as(examples)


def quadratic(outputs, labels):
    return (outputs - labels).square().mean() / 2


class QuadraticTrainer(feddyn_trainer.Trainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        self.loss_strategy.base_loss_fn = quadratic


def configuration(mode="uniform", counts=(2, 2), spawn=False, population=2):
    result = tomllib.loads((EXAMPLE / "feddyn_MNIST_lenet5.toml").read_text())
    result["clients"].update(total_clients=population, per_round=1)
    result["server"].update(do_test=False, random_seed=17)
    result["trainer"].update(batch_size=100, epochs=2, model_name="org/model")
    result["trainer"].pop("target_accuracy")
    if not spawn:
        result["trainer"].pop("max_concurrency")
    result["parameters"]["optimizer"].update(lr=0.1, momentum=0, weight_decay=0)
    result["algorithm"].update(alpha_coef=0.1, feddyn_weighting=mode)
    if mode == "sample":
        result["algorithm"]["feddyn_sample_counts"] = list(counts)
    return result


def server():
    feddyn_server = load_example("feddyn_server")
    s = feddyn_server.Server(model=ScalarModel, trainer=ComposableTrainer)
    s.init_trainer()
    s.context.trainer, s.context.algorithm = s.trainer, s.algorithm
    s._ensure_session()
    return s


def client(client_id, trainer=QuadraticTrainer):
    c = feddyn_client.create_client(model=ScalarModel, trainer=trainer)
    c.client_id = c._context.client_id = client_id
    c.configure()
    return c


def dispatch(s, ids):
    s.current_round = s.committed_round + 1
    s.selected_clients = list(ids)
    result = {}
    for i in ids:
        s.selected_client_id = i
        response = s.customize_server_response(
            {"id": i, "current_round": s.current_round}, i
        )
        result[i] = (
            response,
            s.customize_server_payload(s.algorithm.extract_weights()),
        )
    return result


class Capture:
    async def send_report_and_payload(self, context, report, payload):
        self.report, self.payload = report, deserialize_tree(payload)


def local(c, assignment, target, count):
    response, payload = assignment
    i, round_id = response["id"], response["current_round"]
    c.client_id = c._context.client_id = i
    c.current_round = c._context.current_round = round_id
    c.configure()
    context = c._context
    c.lifecycle_strategy.process_server_response(context, response)
    context.trainset = TensorDataset(
        torch.ones(count, 1).double(),
        torch.full((count, 1), target, dtype=torch.double),
    )
    context.sampler = torch.utils.data.SequentialSampler(context.trainset)
    capture = Capture()
    asyncio.run(
        c.payload_strategy.handle_server_payload(
            context,
            serialize_tree(payload),
            training=c.training_strategy,
            reporting=c.reporting_strategy,
            communication=capture,
        )
    )
    assert capture.report.num_samples == count
    assert capture.payload[1]["num_samples"] == count
    return SimpleNamespace(report=capture.report, payload=capture.payload)


def rational_round(x, histories, selected, targets, counts, mode):
    """No Plato helper: independently differentiate the published equations."""
    n, alpha, eta = len(histories), Fraction(1, 10), Fraction(1, 10)
    endpoints, next_histories = {}, list(histories)
    for i in selected:
        a = alpha if mode == "uniform" else alpha * sum(counts) / (n * counts[i - 1])
        y = x
        for _ in range(2):
            y -= eta * (y - targets[i - 1] + a * (y - x + histories[i - 1]))
        endpoints[i] = y
        next_histories[i - 1] += y - x
    return (
        sum(endpoints.values()) / len(selected) + sum(next_histories) / n,
        next_histories,
        endpoints,
    )


@pytest.mark.parametrize(
    "defect",
    [
        "exit",
        "save",
        "missing",
        "corrupt",
        "token",
        "client",
        "round",
        "key",
        "shape",
        "nan",
        "equation",
        "model",
        "parent",
    ],
)
@pytest.mark.slow
def test_actual_spawned_rejection_retry_and_logical_reuse(tmp_path, defect):
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        "-m",
        "tests.integration.feddyn_rejection_worker",
        str(tmp_path / "runtime"),
        str(output),
        defect,
    ]
    completed = subprocess.run(
        ["zsh", "-lc", shlex.join(command)], capture_output=True, text=True, timeout=100
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(output.read_text()) == pytest.approx(
        [1.433, 1.9717445, 1.47705273425], abs=1e-12
    )


def run_partial(
    root,
    mode="uniform",
    counts=(2, 2),
    spawn=False,
    reuse=True,
    model_name="org/model",
    visits=(1, 2, 1),
):
    records = []
    config = configuration(mode, counts, spawn)
    config["trainer"]["model_name"] = model_name
    with configure_environment(config, runtime_root=root):
        s = server()
        c = client(1)
        x, histories = Fraction(2), [Fraction(0), Fraction(0)]
        for i in visits:
            assignment = dispatch(s, [i])[i]
            if not reuse:
                c = client(i)
            update = local(c, assignment, (0.0, 4.0)[i - 1], counts[i - 1])
            next_x, next_histories, endpoints = rational_round(
                x, histories, [i], [Fraction(0), Fraction(4)], list(counts), mode
            )
            assert update.payload[0]["theta"].item() == pytest.approx(
                float(endpoints[i]), abs=1e-12
            )
            strategy = c.trainer.model_update_strategy
            assert strategy.result["history"]["theta"].item() == pytest.approx(
                float(next_histories[i - 1]), abs=1e-12
            )
            assert strategy.result["completed_steps"] == 2
            s.updates = [update]
            asyncio.run(s._process_reports())
            assert s.trainer.model.theta.item() == pytest.approx(
                float(next_x), abs=1e-12
            )
            for j, expected in enumerate(next_histories, 1):
                assert s.histories[j]["theta"].item() == pytest.approx(
                    float(expected), abs=1e-12
                )
            s.save_to_checkpoint()
            assert s.dispatches == {} and s._pending is None
            records.append(
                dict(
                    client=i,
                    y=float(endpoints[i]),
                    x=float(next_x),
                    histories=list(map(float, next_histories)),
                )
            )
            x, histories = next_x, next_histories
        assert not list((root / "models").glob("feddyn_grad*"))
    return records


@pytest.mark.parametrize("mode,counts", [("uniform", (2, 2)), ("sample", (1, 3))])
@pytest.mark.parametrize(
    "spawn", [False, pytest.param(True, marks=pytest.mark.slow)]
)
def test_consecutive_same_client_uses_current_cloud_and_dispatched_history(
    tmp_path, mode, counts, spawn
):
    if not spawn:
        run_partial(tmp_path, mode, counts, visits=(1, 1, 2, 1))
        return
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        "-m",
        "tests.integration.feddyn_round_worker",
        str(tmp_path / "runtime"),
        str(output),
        mode,
        ",".join(map(str, counts)),
        "1,1,2,1",
    ]
    completed = subprocess.run(
        ["zsh", "-lc", shlex.join(command)], capture_output=True, text=True, timeout=100
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert [r["client"] for r in json.loads(output.read_text())] == [1, 1, 2, 1]


@pytest.mark.parametrize(
    "mode,counts",
    [
        ("uniform", (2, 2)),
        ("sample", (1, 3)),
        ("sample", (2, 2)),
        ("uniform", (17, 93)),
    ],
)
@pytest.mark.parametrize("reuse", [False, True])
def test_three_round_actual_partial_participation(tmp_path, mode, counts, reuse):
    run_partial(tmp_path, mode, counts, reuse=reuse)


@pytest.mark.parametrize("mode", ["uniform", "sample"])
def test_multiple_participants_preserve_inactive_history_population_mean(
    tmp_path, mode
):
    config = configuration(mode, (1, 3, 2), population=3)
    config["clients"]["per_round"] = 2
    with configure_environment(config, runtime_root=tmp_path):
        s = server()
        for i, value in enumerate((0.4, -0.2, 0.6), 1):
            s.histories[i]["theta"].fill_(value)
        assignments = dispatch(s, [1, 3])
        updates = [
            local(client(i), assignments[i], 0, (1, 3, 2)[i - 1]) for i in (1, 3)
        ]
        # Aggregation-only independent endpoints; metadata still comes from
        # real successful local training with the actual configured count.
        for u, y in zip(updates, (1.0, 4.0)):
            u.payload[0]["theta"].fill_(y)
        s.updates = updates
        asyncio.run(s._process_reports())
        assert s.trainer.model.theta.item() == pytest.approx(3.1, abs=1e-12)
        assert [s.histories[i]["theta"].item() for i in (1, 2, 3)] == pytest.approx(
            [-0.6, -0.2, 2.6]
        )


@pytest.mark.parametrize("mode", ["uniform", "sample"])
def test_actual_full_participation_first_cloud(tmp_path, mode):
    config = configuration(mode, (1, 3))
    config["clients"]["per_round"] = 2
    with configure_environment(config, runtime_root=tmp_path):
        s = server()
        assignments = dispatch(s, [1, 2])
        s.updates = [
            local(client(i), assignments[i], (0.0, 4.0)[i - 1], (1, 3)[i - 1])
            for i in (1, 2)
        ]
        asyncio.run(s._process_reports())
        assert s.trainer.model.theta.item() == pytest.approx(
            2.0 if mode == "uniform" else 2.0026666666666667, abs=1e-12
        )


@pytest.mark.parametrize("mode,counts", [("uniform", (2, 2)), ("sample", (1, 3))])
@pytest.mark.slow
def test_actual_spawn_reused_client_matches_partial_oracle(tmp_path, mode, counts):
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        "-m",
        "tests.integration.feddyn_round_worker",
        str(tmp_path / "runtime"),
        str(output),
        mode,
        ",".join(map(str, counts)),
    ]
    completed = subprocess.run(
        ["zsh", "-lc", shlex.join(command)], capture_output=True, text=True, timeout=90
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(json.loads(output.read_text())) == 3


def committed_state(s):
    """Compare known state without importing production arithmetic helpers."""
    return copy.deepcopy(
        (
            s.trainer.model.state_dict(),
            s.histories,
            s.observed_counts,
            s.committed_round,
            s.accepted_tokens,
            s.dispatches,
        )
    )


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            assert_state_equal(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def bad_payload(update, mode):
    model, meta = update.payload
    if mode == "missing":
        model.clear()
    elif mode == "extra":
        model["extra"] = torch.zeros(1).double()
    elif mode == "shape":
        model["theta"] = model["theta"].squeeze()
    elif mode == "dtype":
        model["theta"] = model["theta"].float()
    elif mode in ("nan", "inf"):
        model["theta"].fill_(float(mode))
    elif mode == "non-tensor":
        model["theta"] = [1.0]
    elif mode == "token":
        meta["dispatch_token"] = "0" * 32
    elif mode == "stale-round":
        meta["round"] = 100
    elif mode == "unknown-client":
        meta["client_id"] = 99
    elif mode == "report-id":
        update.report.client_id = 99
    elif mode == "count":
        meta["num_samples"] = 0
    elif mode == "report-count":
        update.report.num_samples = 0
    elif mode == "fractional-count":
        meta["num_samples"] = 2.0
    elif mode == "bool-count":
        meta["num_samples"] = True
    elif mode == "steps":
        meta["completed_steps"] = 0
    elif mode == "version":
        meta["version"] = True


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "extra",
        "shape",
        "dtype",
        "nan",
        "inf",
        "non-tensor",
        "token",
        "stale-round",
        "unknown-client",
        "report-id",
        "count",
        "report-count",
        "fractional-count",
        "bool-count",
        "steps",
        "version",
        "duplicate",
        "incomplete",
        "empty",
    ],
)
@pytest.mark.parametrize("position", [0, 1])
def test_whole_batch_rejection_preserves_every_history_and_retry_identity(
    tmp_path, mode, position
):
    config = configuration()
    config["clients"]["per_round"] = 2
    with configure_environment(config, runtime_root=tmp_path):
        s = server()
        assignments = dispatch(s, [1, 2])
        valid = [local(client(i), assignments[i], (0.0, 4.0)[i - 1], 2) for i in (1, 2)]
        s.updates = copy.deepcopy(valid)
        original = committed_state(s)
        s.save_to_checkpoint()
        before_bytes = Path(s.checkpoint_bundle_path()).read_bytes()
        if mode == "duplicate":
            s.updates[position] = copy.deepcopy(s.updates[1 - position])
        elif mode == "incomplete":
            s.updates.pop(position)
        elif mode == "empty":
            s.updates = []
        else:
            bad_payload(s.updates[position], mode)
        raw = serialize_tree([u.payload for u in s.updates])
        with pytest.raises((ValueError, TypeError)):
            asyncio.run(s._process_reports())
        assert_state_equal(committed_state(s), original)
        assert Path(s.checkpoint_bundle_path()).read_bytes() == before_bytes
        assert s._pending is None
        assert serialize_tree([u.payload for u in s.updates]) == raw
        s.updates = valid
        asyncio.run(s._process_reports())
        assert s.trainer.model.theta.item() == pytest.approx(2.0, abs=1e-12)
        assert s.committed_round == 1
        with pytest.raises(ValueError):
            asyncio.run(s._process_reports())


class InterruptCallbacks(ServerCallback):
    def __init__(self, mode):
        self.mode = mode

    def on_weights_received(self, server, weights_received):
        s, payloads = server, weights_received
        if self.mode == "receive-error":
            raise RuntimeError("Deliberate receive failure")
        if self.mode == "receive-model":
            payloads[-1][0]["theta"].fill_(float("nan"))
        if self.mode == "receive-count":
            s.updates[-1].report.num_samples = -1
        if self.mode == "receive-history":
            s.histories[2]["theta"].add_(1)
        if self.mode == "receive-dispatch":
            s.dispatches[1]["baseline"]["theta"].add_(1)
        if self.mode == "receive-round":
            s.current_round += 1

    def on_weights_aggregated(self, server, updates):
        s = server
        if self.mode == "aggregate-error":
            raise RuntimeError("Deliberate aggregation callback failure")
        if self.mode == "aggregate-model":
            s.trainer.model.theta.data.add_(1)
        if self.mode == "aggregate-history":
            s.histories[1]["theta"].add_(1)

    def on_clients_processed(self, server, **kwargs):
        if self.mode == "postcommit":
            raise RuntimeError("Deliberate reporting failure")


@pytest.mark.parametrize(
    "mode",
    [
        "receive-error",
        "receive-model",
        "receive-count",
        "receive-history",
        "receive-dispatch",
        "receive-round",
        "aggregate-error",
        "aggregate-model",
        "aggregate-history",
        "partial-load",
        "arithmetic",
        "postcommit",
    ],
)
def test_callback_and_partial_load_boundary(tmp_path, monkeypatch, mode):
    with configure_environment(configuration(), runtime_root=tmp_path):
        s = server()
        assignment = dispatch(s, [1])[1]
        valid = local(client(1), assignment, 0, 2)
        s.updates = [copy.deepcopy(valid)]
        callback = InterruptCallbacks(mode)
        s.callback_handler.add_callback(callback)
        original = committed_state(s)
        load = s.algorithm.load_weights
        aggregate = s.aggregate_weights
        if mode == "partial-load":

            def broken_load(weights):
                load(weights)
                raise RuntimeError("Deliberate partial model load")

            monkeypatch.setattr(s.algorithm, "load_weights", broken_load)
        if mode == "arithmetic":

            async def broken_aggregate(*args):
                raise RuntimeError("Deliberate arithmetic failure")

            monkeypatch.setattr(s, "aggregate_weights", broken_aggregate)
        with pytest.raises((ValueError, RuntimeError)):
            asyncio.run(s._process_reports())
        assert s._pending is None
        if mode == "postcommit":
            assert s.committed_round == 1
            assert s.trainer.model.theta.item() == pytest.approx(1.433, abs=1e-12)
            return
        assert_state_equal(committed_state(s), original)
        s.callback_handler.callbacks.remove(callback)
        monkeypatch.setattr(s.algorithm, "load_weights", load)
        monkeypatch.setattr(s, "aggregate_weights", aggregate)
        s.updates = [valid]
        asyncio.run(s._process_reports())
        assert s.trainer.model.theta.item() == pytest.approx(1.433, abs=1e-12)


def zero_task(outputs, labels):
    return outputs.sum() * 0


class ZeroTrainer(QuadraticTrainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        self.loss_strategy.base_loss_fn = zero_task


class ObserveSteps(TrainerCallback):
    def __init__(self):
        self.steps = 0

    def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
        self.steps += 1


class SkippedSteps(DefaultTrainingStepStrategy):
    def __init__(self, skip):
        super().__init__()
        self.skip = skip

    def training_step(
        self, model, optimizer, examples, labels, loss_criterion, context
    ):
        if self.skip:
            self.skip -= 1
            optimizer.zero_grad(set_to_none=True)
            context.state["optimizer_step_completed"] = False
            return loss_criterion(model(examples), labels).detach()
        return super().training_step(
            model, optimizer, examples, labels, loss_criterion, context
        )


@pytest.mark.parametrize(
    "window,batches,skip,steps",
    [(2, 3, 0, 2), (2, 4, 0, 2), (5, 3, 0, 1), (1, 2, 1, 1)],
)
def test_accumulated_tail_and_skips_count_only_real_updates(
    tmp_path, monkeypatch, window, batches, skip, steps
):
    config = configuration()
    config["trainer"].update(gradient_accumulation_steps=window, batch_size=1, epochs=1)
    with configure_environment(config, runtime_root=tmp_path):
        s = server()
        s.histories[1]["theta"].fill_(0.5)
        assignment = dispatch(s, [1])[1]
        c = client(1, ZeroTrainer)
        trainer = c.trainer
        if skip:
            trainer.training_step_strategy = SkippedSteps(skip)
        observed = ObserveSteps()
        trainer.callback_handler.add_callback(observed)
        counts = dict(optimizer=0, update=0, history=0)
        optimizer_hook = trainer.optimizer_strategy.on_optimizer_step
        update_hook = trainer.model_update_strategy.after_step
        history_hook = trainer.model_update_strategy.on_train_end

        def optimizer_completed(*args):
            counts["optimizer"] += 1
            return optimizer_hook(*args)

        def update_completed(*args):
            counts["update"] += 1
            return update_hook(*args)

        def history_completed(*args):
            counts["history"] += 1
            return history_hook(*args)

        monkeypatch.setattr(
            trainer.optimizer_strategy, "on_optimizer_step", optimizer_completed
        )
        monkeypatch.setattr(
            trainer.model_update_strategy, "after_step", update_completed
        )
        monkeypatch.setattr(
            trainer.model_update_strategy, "on_train_end", history_completed
        )
        local(c, assignment, 0, batches)
        expected = Fraction(2)
        for _ in range(steps):
            expected -= Fraction(1, 100) * (expected - 2 + Fraction(1, 2))
        assert trainer.model.theta.item() == pytest.approx(float(expected), abs=1e-12)
        assert trainer.model_update_strategy.result["history"][
            "theta"
        ].item() == pytest.approx(float(expected - Fraction(3, 2)), abs=1e-12)
        assert counts == dict(optimizer=steps, update=steps, history=1)
        assert observed.steps == steps
        assert trainer.lr_scheduler is None


@pytest.mark.parametrize("empty", [False, True])
def test_empty_or_all_skipped_attempt_refuses_outbound_and_valid_retry(tmp_path, empty):
    config = configuration()
    config["trainer"].update(epochs=1)
    with configure_environment(config, runtime_root=tmp_path):
        s = server()
        assignment = dispatch(s, [1])[1]
        c = client(1, ZeroTrainer)
        c.trainer.training_step_strategy = SkippedSteps(100)
        before = committed_state(s)
        with pytest.raises(ValueError):
            local(c, assignment, 0, 0 if empty else 2)
        assert c.trainer.model.theta.item() == 2
        assert not c.trainer.model_update_strategy.accepted
        assert c.trainer.model.theta.grad is None
        assert_state_equal(committed_state(s), before)
        c.trainer.training_step_strategy = DefaultTrainingStepStrategy()
        update = local(c, assignment, 0, 2)
        assert update.payload[1]["completed_steps"] == 1


class MixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.double))
        self.b = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.double))
        self.frozen = torch.nn.Parameter(
            torch.tensor([4.0], dtype=torch.double), requires_grad=False
        )
        self.register_buffer("floating", torch.tensor([0.25], dtype=torch.double))
        self.register_buffer("integral", torch.tensor([7], dtype=torch.int64))
        self.register_buffer("boolean", torch.tensor([True]))

    def forward(self, x):
        return self.a * x + self.b


@pytest.mark.parametrize(
    "field",
    [
        "flag",
        "shape",
        "frozen",
        "floating",
        "integral",
        "boolean",
        "nan-buffer",
        "count",
    ],
)
def test_actual_local_schema_buffer_and_count_mutation_rolls_back(tmp_path, field):
    with configure_environment(configuration(), runtime_root=tmp_path):
        module = load_example("feddyn_server")
        s = module.Server(model=MixedModel, trainer=ComposableTrainer)
        s.init_trainer()
        s._ensure_session()
        c = feddyn_client.create_client(model=MixedModel, trainer=QuadraticTrainer)
        c.client_id = c._context.client_id = 1
        c.configure()
        baseline = copy.deepcopy(c.trainer.model.state_dict())
        identities = {n: id(p) for n, p in c.trainer.model.named_parameters()}

        class Mutate(TrainerCallback):
            def on_train_step_end(self, trainer, config, batch, loss, **kwargs):
                if field == "flag":
                    trainer.model.a.requires_grad_(False)
                elif field == "shape":
                    trainer.model.a.data = torch.ones(2, dtype=torch.double)
                elif field == "count":
                    trainer.context.state["train_loader"].sampler.data_source = list(
                        range(3)
                    )
                elif field == "boolean":
                    trainer.model.boolean.logical_not_()
                elif field == "nan-buffer":
                    trainer.model.floating.fill_(float("nan"))
                else:
                    getattr(trainer.model, field).data.add_(1)

        mutate = Mutate()
        c.trainer.callback_handler.add_callback(mutate)
        assignment = dispatch(s, [1])[1]
        with pytest.raises(ValueError):
            local(c, assignment, 0, 2)
        assert_state_equal(c.trainer.model.state_dict(), baseline)
        assert identities == {n: id(p) for n, p in c.trainer.model.named_parameters()}
        assert c.trainer.model.a.requires_grad
        assert not c.trainer.model_update_strategy.accepted
        c.trainer.callback_handler.callbacks.remove(mutate)
        update = local(c, assignment, 0, 2)
        s.updates = [update]
        asyncio.run(s._process_reports())
        assert s.committed_round == 1


@pytest.mark.parametrize(
    "defect",
    [
        "model-key",
        "model-extra",
        "model-shape",
        "model-dtype",
        "model-nan",
        "model-inf",
        "model-type",
        "history-key",
        "history-extra",
        "history-shape",
        "history-nan",
        "history-dtype",
        "version",
        "run",
        "round",
        "client",
        "token",
        "mode",
        "alpha",
        "effective",
        "count",
        "header",
    ],
)
def test_actual_inbound_whole_dispatch_validation_before_mutation(tmp_path, defect):
    with configure_environment(configuration(), runtime_root=tmp_path):
        s, c = server(), client(1)
        response, payload = dispatch(s, [1])[1]
        bad = copy.deepcopy(payload)
        model, metadata = bad
        if defect == "model-key":
            model.pop("theta")
        elif defect == "model-extra":
            model["extra"] = torch.zeros(1, dtype=torch.double)
        elif defect == "model-shape":
            model["theta"] = model["theta"].squeeze()
        elif defect == "model-dtype":
            model["theta"] = model["theta"].float()
        elif defect == "model-nan":
            model["theta"].fill_(float("nan"))
        elif defect == "model-inf":
            model["theta"].fill_(float("inf"))
        elif defect == "model-type":
            model["theta"] = [2.0]
        elif defect == "history-key":
            metadata["history"].pop("theta")
        elif defect == "history-extra":
            metadata["history"]["extra"] = torch.zeros(1, dtype=torch.double)
        elif defect == "history-shape":
            metadata["history"]["theta"] = torch.tensor(0.0, dtype=torch.double)
        elif defect == "history-nan":
            metadata["history"]["theta"].fill_(float("nan"))
        elif defect == "history-dtype":
            metadata["history"]["theta"] = metadata["history"]["theta"].float()
        elif defect == "version":
            metadata["version"] = True
        elif defect == "run":
            metadata["run_id"] = "missing"
        elif defect == "round":
            metadata["round"] += 1
        elif defect == "client":
            metadata["client_id"] = 2
        elif defect == "token":
            metadata["dispatch_token"] = "a" * 32
        elif defect == "mode":
            metadata["weighting"] = "sample"
        elif defect == "alpha":
            metadata["base_alpha"] = 0.2
        elif defect == "effective":
            metadata["effective_alpha"] = float("nan")
        elif defect == "count":
            metadata["expected_count_or_null"] = False
        elif defect == "header":
            response["feddyn"]["round"] += 1
        before_model = copy.deepcopy(c.trainer.model.state_dict())
        before_state = copy.deepcopy(c.trainer.model_update_strategy.__dict__)
        with pytest.raises(ValueError):
            local(c, (response, bad), 0, 2)
        assert_state_equal(c.trainer.model.state_dict(), before_model)
        assert_state_equal(c.trainer.model_update_strategy.__dict__, before_state)
        assert not c.trainer.model_update_strategy.accepted
        response = s.customize_server_response({"id": 1, "current_round": 1}, 1)
        update = local(c, (response, payload), 0, 2)
        assert update.payload[0]["theta"].item() == pytest.approx(1.622, abs=1e-12)


@pytest.mark.parametrize("declared", [1, 3])
def test_realized_partition_count_uses_sampler_not_global_backing_data(
    tmp_path, declared
):
    config = configuration("sample", (3, 7))
    with configure_environment(config, runtime_root=tmp_path):
        s, c = server(), client(1)
        response, payload = dispatch(s, [1])[1]
        c.current_round = c._context.current_round = 1
        c.trainer.current_round = 1
        c.lifecycle_strategy.process_server_response(c._context, response)
        c.training_strategy.load_payload(c._context, payload)
        data = TensorDataset(torch.ones(100, 1).double(), torch.zeros(100, 1).double())

        class Partition:
            def get(self):
                return [0, 1, 2]

            def num_samples(self):
                return declared

        if declared != 3:
            with pytest.raises(ValueError, match="count"):
                c.trainer.train(data, Partition())
            assert c.trainer.model.theta.item() == 2
            assert not c.trainer.model_update_strategy.accepted
        else:
            c.trainer.train(data, Partition())
            assert c.trainer.model_update_strategy.result["num_samples"] == 3
            assert c.trainer.loss_strategy._alpha_for_run == pytest.approx(1 / 6)


@pytest.mark.parametrize("defect", ["identity", "schema", "rng", "install", "none"])
def test_pending_resume_handoff_is_atomic_and_consumed_once(
    tmp_path, monkeypatch, defect
):
    from plato.servers import base as server_base

    with configure_environment(configuration(), runtime_root=tmp_path):
        s = server()
        s.save_to_checkpoint()
        s._resume_from_checkpoint()
        pending = s._pending_resume_rng
        if defect == "identity":
            pending["run_id"] = "f" * 32
        elif defect == "schema":
            pending["schema"]["theta"]["shape"] = []
        elif defect == "rng":
            pending["rng"]["torch"] = torch.zeros(3, dtype=torch.uint8)
        before_pending, before_rng = copy.deepcopy(pending), s._rng_snapshot()
        starts = []

        def start(*args, **kwargs):
            starts.append(s._rng_snapshot())
            return "registered"

        monkeypatch.setattr(server_base.Server, "start", start)
        if defect == "install":
            original_set = torch.set_rng_state
            attempted = []

            def fail_once(state):
                attempted.append(1)
                if len(attempted) == 1:
                    raise RuntimeError("Deliberate CPU RNG install failure")
                original_set(state)

            monkeypatch.setattr(torch, "set_rng_state", fail_once)
        if defect != "none":
            with pytest.raises((ValueError, RuntimeError)):
                s.start()
            assert not starts
            assert_state_equal(s._rng_snapshot(), before_rng)
            assert_state_equal(s._pending_resume_rng, before_pending)
        else:
            assert s.start() == "registered"
            assert s._pending_resume_rng is None
            assert_state_equal(starts[0], pending["rng"])
            random.random()
            changed = s._rng_snapshot()
            assert s.start() == "registered"
            assert_state_equal(starts[1], changed)


@pytest.mark.parametrize(
    "ownership", ["reverse", "missing", "duplicate", "foreign", "frozen", "adam"]
)
def test_optimizer_parameter_identity_and_static_full_state(
    tmp_path, monkeypatch, ownership
):
    from plato.trainers.strategies.algorithms.feddyn_strategy import model_schema

    with configure_environment(configuration(), runtime_root=tmp_path):
        s = load_example("feddyn_server").Server(
            model=MixedModel, trainer=ComposableTrainer
        )
        s.init_trainer()
        s._ensure_session()
        assignments = dispatch(s, [1])
        c = feddyn_client.create_client(model=MixedModel, trainer=QuadraticTrainer)
        c.client_id = c._context.client_id = 1
        c.configure()
        trainer = c.trainer
        params = [trainer.model.b, trainer.model.a]
        if ownership == "missing":
            params = [trainer.model.a]
        if ownership == "duplicate":
            params = [trainer.model.a, trainer.model.a, trainer.model.b]
        if ownership == "foreign":
            params.append(torch.nn.Parameter(torch.ones(1).double()))
        if ownership == "frozen":
            params.append(trainer.model.frozen)
        cls = torch.optim.Adam if ownership == "adam" else torch.optim.SGD
        if ownership == "duplicate":
            with pytest.warns(UserWarning, match="duplicate"):
                optimizer = cls(params, lr=0.1)
        else:
            optimizer = cls(params, lr=0.1)
        monkeypatch.setattr(
            trainer.optimizer_strategy, "create_optimizer", lambda *args: optimizer
        )
        before = copy.deepcopy(trainer.model.state_dict())
        identities = [id(p) for p in trainer.model.parameters()]
        if ownership != "reverse":
            with pytest.raises(ValueError, match="FedDyn"):
                local(c, assignments[1], 0, 2)
            assert_state_equal(trainer.model.state_dict(), before)
            assert not trainer.model_update_strategy.accepted
            return
        update = local(c, assignments[1], 0, 2)
        # Independent two steps: a=b each receive gradient a+b plus shifted
        # proximal derivative; at initialization5 then3.95.
        assert trainer.model.a.item() == pytest.approx(1.105, abs=1e-12)
        assert trainer.model.b.item() == pytest.approx(2.105, abs=1e-12)
        assert set(trainer.model_update_strategy.result["history"]) == {"a", "b"}
        assert [id(p) for p in trainer.model.parameters()] == identities
        for k in ("frozen", "floating", "integral", "boolean"):
            assert torch.equal(update.payload[0][k], before[k])
        s.updates = [update]
        asyncio.run(s._process_reports())
        assert model_schema(s.trainer.model) == s.schema
        for k in ("frozen", "floating", "integral", "boolean"):
            assert torch.equal(s.trainer.model.state_dict()[k], before[k])
        s.save_to_checkpoint()


@pytest.mark.parametrize("mode,counts", [("uniform", (2, 2)), ("sample", (1, 3))])
@pytest.mark.parametrize("name", ["toy", "org/model", "x" * 240])
def test_complete_bundle_round_two_to_three_matches_rational_reference(
    tmp_path, mode, counts, name
):
    config = configuration(mode, counts)
    config["trainer"]["model_name"] = name
    with configure_environment(config, runtime_root=tmp_path):
        s = server()
        x, h = Fraction(2), [Fraction(0), Fraction(0)]
        for i in (1, 2):
            assignment = dispatch(s, [i])[i]
            update = local(client(i), assignment, (0.0, 4.0)[i - 1], counts[i - 1])
            s.updates = [update]
            asyncio.run(s._process_reports())
            x, h, _ = rational_round(
                x, h, [i], [Fraction(0), Fraction(4)], list(counts), mode
            )
        s.save_to_checkpoint()
        saved = torch.load(s.checkpoint_bundle_path(), weights_only=True)
        assert saved["committed_round"] == 2
        old_run = s.run_id
        resumed = server()
        before_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        resumed._resume_from_checkpoint()
        # Loading stages RNG without mutating live generators.
        assert random.getstate() == before_rng[0]
        np.testing.assert_array_equal(np.random.get_state()[1], before_rng[1][1])
        assert torch.equal(torch.get_rng_state(), before_rng[2])
        assert (
            resumed.run_id == old_run
            and resumed.current_round == resumed.committed_round == 2
        )
        assert resumed._pending_resume_rng is not None
        assignment = dispatch(resumed, [1])[1]
        update = local(client(1), assignment, 0, counts[0])
        resumed.updates = [update]
        asyncio.run(resumed._process_reports())
        expected_x, expected_h, _ = rational_round(
            x, h, [1], [Fraction(0), Fraction(4)], list(counts), mode
        )
        assert resumed.trainer.model.theta.item() == pytest.approx(
            float(expected_x), abs=1e-12
        )
        for i, expected in enumerate(expected_h, 1):
            assert resumed.histories[i]["theta"].item() == pytest.approx(
                float(expected), abs=1e-12
            )
        resumed.warm_start_model(resumed.trainer.model.state_dict())
        assert resumed.run_id != old_run and resumed.committed_round == 0
        assert all(
            not torch.count_nonzero(v)
            for entry in resumed.histories.values()
            for v in entry.values()
        )


@pytest.mark.parametrize(
    "bad",
    [
        "version",
        "run",
        "round",
        "config",
        "schema",
        "model",
        "history",
        "count",
        "rng-selection",
        "rng-python",
        "rng-numpy",
        "rng-torch",
        "model-only",
    ],
)
def test_rejected_bundle_preserves_model_history_live_rng_and_previous_pending(
    tmp_path, bad
):
    with configure_environment(configuration(), runtime_root=tmp_path):
        s = server()
        update = local(client(1), dispatch(s, [1])[1], 0, 2)
        s.updates = [update]
        asyncio.run(s._process_reports())
        s.save_to_checkpoint()
        s._resume_from_checkpoint()
        original = committed_state(s)
        pending = copy.deepcopy(s._pending_resume_rng)
        rng = s._rng_snapshot()
        path = Path(s.checkpoint_bundle_path())
        bundle = torch.load(path, weights_only=True)
        if bad == "version":
            bundle["version"] = True
        elif bad == "run":
            bundle["run_id"] = "other-run"
        elif bad == "round":
            bundle["committed_round"] = -1
        elif bad == "config":
            bundle["config"]["base_alpha"] = 0.2
        elif bad == "schema":
            bundle["schema"]["theta"]["trainable"] = False
        elif bad == "model":
            bundle["model"]["theta"].fill_(float("nan"))
        elif bad == "history":
            bundle["histories"][2]["theta"] = torch.tensor(0.0).double()
        elif bad == "count":
            bundle["counts"][1] = True
        elif bad == "rng-selection":
            bundle["rng"]["context_selection"] = random.Random(99).getstate()
        elif bad == "rng-python":
            bundle["rng"]["python"] = (3, (), None)
        elif bad == "rng-numpy":
            bundle["rng"]["numpy"][1] = torch.zeros(3, dtype=torch.uint32)
        elif bad == "rng-torch":
            bundle["rng"]["torch"] = torch.zeros(3, dtype=torch.uint8)
        else:
            bundle = bundle["model"]
        torch.save(bundle, path)
        with pytest.raises(ValueError, match="FedDyn"):
            s._resume_from_checkpoint()
        assert_state_equal(committed_state(s), original)
        assert_state_equal(s._pending_resume_rng, pending)
        assert_state_equal(s._rng_snapshot(), rng)


def test_failed_bundle_serialization_preserves_previous_canonical_bytes(
    tmp_path, monkeypatch
):
    with configure_environment(configuration(), runtime_root=tmp_path):
        s = server()
        s.save_to_checkpoint()
        path = Path(s.checkpoint_bundle_path())
        before = path.read_bytes()

        def failed_save(snapshot, stream):
            stream.write(b"partial")
            raise OSError("Deliberate serialization failure")

        monkeypatch.setattr(torch, "save", failed_save)
        with pytest.raises(OSError, match="serialization"):
            s.save_to_checkpoint()
        assert path.read_bytes() == before
        assert not list(path.parent.glob(".feddyn-*"))
