"""Actual shipped-config SCAFFOLD client/processor/server flow (A02/A05-A08)."""

import asyncio
import copy
import importlib.util
import json
import pickle
import shlex
import subprocess
import sys
import tomllib
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from plato.config import Config
from plato.serialization.safetensor import deserialize_tree, serialize_tree
from plato.trainers.strategies.loss_criterion import DefaultLossCriterionStrategy
from tests.integration.utils import configure_environment, isolated_config_state
from tests.trainers.test_scaffold_strategy import (
    MixedParameters,
    ScalarModel,
    quadratic_loss,
    scalar_controls,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/customized_client_training/scaffold/scaffold_MNIST_lenet5.toml"
)


def load_example(name):
    spec = importlib.util.spec_from_file_location(
        name, CONFIG_PATH.parent / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scaffold_client = load_example("scaffold_client")
scaffold_trainer = load_example("scaffold_trainer")
scaffold_callback = load_example("scaffold_callback")
ScaffoldCallback = scaffold_callback.ScaffoldCallback
ExtractControlVariatesProcessor = scaffold_callback.ExtractControlVariatesProcessor
SendControlVariateProcessor = scaffold_callback.SendControlVariateProcessor


def shipped_config(*, spawn=False, lr=0.1, momentum=0.0):
    config = tomllib.loads(CONFIG_PATH.read_text())
    config["clients"].update(total_clients=3, per_round=2)
    config["server"].update(do_test=False, random_seed=29)
    config["trainer"].update(batch_size=6, epochs=2, model_name="org/model")
    if not spawn:
        config["trainer"].pop("max_concurrency")
    config["parameters"]["optimizer"].update(lr=lr, momentum=momentum)
    return config


class QuadraticTrainer(scaffold_trainer.Trainer):
    """Inject the bounded scalar/loss oracle without replacing the optimizer."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        self.loss_strategy = DefaultLossCriterionStrategy(quadratic_loss)
        self.loss_strategy.setup(self.context)


class MissingStateTrainer(QuadraticTrainer):
    def train_process(self, config, trainset, sampler, **kwargs):
        super().train_process(config, trainset, sampler, **kwargs)
        Path(self._training_state_path(config["run_id"])).unlink()


class StaleStateTrainer(QuadraticTrainer):
    def train_process(self, config, trainset, sampler, **kwargs):
        super().train_process(config, trainset, sampler, **kwargs)
        path = Path(self._training_state_path(config["run_id"]))
        state = pickle.loads(path.read_bytes())
        state["token"] = "previous-attempt"
        path.write_bytes(pickle.dumps(state))


class CaptureResult:
    async def send_report_and_payload(self, context, report, payload):
        self.report, self.payload = report, deserialize_tree(payload)


def create_client(client_id):
    client = scaffold_client.create_client(
        model=ScalarModel, trainer=QuadraticTrainer, callbacks=[ScaffoldCallback]
    )
    client.client_id = client._context.client_id = client_id
    client.configure()  # Assign identity before the inbound callback/processor.
    return client


def local_round(client, payload, *, round_id, target, samples):
    context = client._context
    context.current_round = client.current_round = round_id
    context.trainset = TensorDataset(
        torch.ones(samples, 1).double(),
        torch.full((samples, 1), target, dtype=torch.double),
    )
    context.sampler = torch.utils.data.SequentialSampler(context.trainset)
    capture = CaptureResult()
    asyncio.run(
        client.payload_strategy.handle_server_payload(
            context,
            serialize_tree(payload),
            training=client.training_strategy,
            reporting=client.reporting_strategy,
            communication=capture,
        )
    )
    assert capture.report.num_samples == samples
    strategy = client.trainer.model_update_strategy
    assert strategy.local_steps == 2
    return capture.report, capture.payload


def make_server(model=ScalarModel):
    server = load_example("scaffold_server").Server(
        model=model, trainer=QuadraticTrainer
    )
    server.init_trainer()
    server.context.trainer, server.context.algorithm = server.trainer, server.algorithm
    return server


def aggregate(server, received):
    reports, payloads = zip(*received)
    updates = [
        SimpleNamespace(report=report, payload=payload) for report, payload in received
    ]
    weights = server.weights_received(list(payloads))
    updated = asyncio.run(
        server.aggregation_strategy.aggregate_weights(
            updates, server.algorithm.extract_weights(), weights, server.context
        )
    )
    server.algorithm.load_weights(updated)
    server.weights_aggregated(updates)
    return server.customize_server_payload(server.algorithm.extract_weights())


def run_two_round_scenario(root, *, spawn=False):
    """Return actual results to the bounded subprocess evidence harness."""
    config = shipped_config(spawn=spawn)
    records = []
    with configure_environment(config, runtime_root=root):
        torch.manual_seed(29)
        server = make_server()
        server.server_control_variate = scalar_controls(2.0)
        clients = {client_id: create_client(client_id) for client_id in (1, 2, 3)}
        for client_id, ci in ((1, 1.0), (2, 3.0), (3, 2.0)):
            clients[
                client_id
            ].trainer.model_update_strategy.client_control_variate = scalar_controls(ci)
        expected_rounds = [
            ((1, 2), [0.62, 1.38], [0.9, -0.9], [-0.1, -3.9], 1.19, 2 / 3),
            (
                (1, 3),
                [30247 / 30000, 30817 / 30000],
                [6853 / 6000, 12883 / 6000],
                [1453 / 6000, 883 / 6000],
                10209 / 10000,
                896 / 1125,
            ),
        ]
        counts, targets = {1: 2, 2: 6, 3: 4}, {1: 0.0, 2: 2.0, 3: -1.0}
        for round_id, (selected, ys, cis, deltas, global_x, global_c) in enumerate(
            expected_rounds, 1
        ):
            payload = server.customize_server_payload(
                server.algorithm.extract_weights()
            )
            received, local_results = [], []
            inactive_before = {
                client_id: copy.deepcopy(
                    client.trainer.model_update_strategy.client_control_variate
                )
                for client_id, client in clients.items()
                if client_id not in selected
            }
            for index, client_id in enumerate(selected):
                client = clients[client_id]
                received.append(
                    local_round(
                        client,
                        payload,
                        round_id=round_id,
                        target=targets[client_id],
                        samples=counts[client_id],
                    )
                )
                strategy = client.trainer.model_update_strategy
                actual = [
                    client.trainer.model.theta.item(),
                    strategy.client_control_variate["theta"].item(),
                    received[-1][1][1]["theta"].item(),
                ]
                assert actual == pytest.approx(
                    [ys[index], cis[index], deltas[index]], abs=1e-12
                )
                if not spawn:
                    assert strategy.server_control_variate[
                        "theta"
                    ].item() == pytest.approx(payload[1]["theta"].item())
                local_results.append(
                    {
                        "client": client_id,
                        "y_ci_delta": actual,
                        "samples": received[-1][0].num_samples,
                    }
                )
            next_payload = aggregate(server, received)
            assert server.trainer.model.theta.item() == pytest.approx(
                global_x, abs=1e-12
            )
            assert next_payload[1]["theta"].item() == pytest.approx(global_c, abs=1e-12)
            all_ci = [
                client.trainer.model_update_strategy.client_control_variate[
                    "theta"
                ].item()
                for client in clients.values()
            ]
            assert sum(all_ci) / 3 == pytest.approx(global_c, abs=1e-12)
            for client_id, controls in inactive_before.items():
                torch.testing.assert_close(
                    clients[
                        client_id
                    ].trainer.model_update_strategy.client_control_variate["theta"],
                    controls["theta"],
                )
            records.append(
                {
                    "round": round_id,
                    "locals": local_results,
                    "global_x": global_x,
                    "global_c": global_c,
                    "all_ci": all_ci,
                }
            )
        assert not list((root / "models").glob("*.train.pkl")) if spawn else True
    return {
        "spawn": spawn,
        "config_source": str(CONFIG_PATH),
        "explicit_overrides": config,
        "rounds": records,
    }


def test_shipped_configuration_two_round_independent_rational_oracle(tmp_path):
    run_two_round_scenario(tmp_path)


def test_shipped_spawn_pipeline_parent_control_return_with_deadline(tmp_path):
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        "-m",
        "tests.integration.scaffold_round_worker",
        str(tmp_path / "runtime"),
        str(output),
    ]
    process = subprocess.run(
        ["zsh", "-lc", shlex.join(command)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(output.read_text())
    assert result["spawn"] is True
    assert len(result["rounds"]) == 2
    assert result["rounds"][1]["global_x"] == pytest.approx(10209 / 10000)


def run_failed_worker_scenario(root, mode):
    with configure_environment(shipped_config(spawn=True), runtime_root=root):
        client = create_client(1)
        client.trainer.model_update_strategy.client_control_variate = scalar_controls(
            1.0
        )
        local_round(
            client,
            [scalar_controls(1.0), scalar_controls(2.0)],
            round_id=1,
            target=0.0,
            samples=2,
        )
        client.trainer.__class__ = (
            MissingStateTrainer if mode == "missing" else StaleStateTrainer
        )
        with pytest.raises(RuntimeError, match="successful"):
            local_round(
                client,
                [scalar_controls(1.0), scalar_controls(3.0)],
                round_id=2,
                target=0.0,
                samples=2,
            )
        # The default client records its training error; outbound processing must
        # refuse the failed worker result instead of sending the old delta.
        assert client.trainer.context.state.get("client_control_variate_delta") is None
        assert isinstance(client._context.state.get("training_error"), ValueError)
        with pytest.raises(RuntimeError, match="successful"):
            SendControlVariateProcessor(1, client.trainer).process(scalar_controls(1.0))
    return {"failure_mode": mode, "outbound_refused": True}


@pytest.mark.parametrize("mode", ["missing", "stale"])
def test_actual_failed_worker_handoff_refuses_previous_delta(tmp_path, mode):
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        "-m",
        "tests.integration.scaffold_round_worker",
        str(tmp_path / "runtime"),
        str(output),
        mode,
    ]
    process = subprocess.run(
        ["zsh", "-lc", shlex.join(command)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(output.read_text())["outbound_refused"] is True


def test_reused_and_reconstructed_clients_match_dedicated_two_round_controls(tmp_path):
    with configure_environment(shipped_config(), runtime_root=tmp_path):
        reused = create_client(1)
        dedicated = {client_id: create_client(client_id) for client_id in (1, 2)}
        for client_id, client in dedicated.items():
            strategy = client.trainer.model_update_strategy
            strategy.save_path = str(tmp_path / f"reference-{client_id}")
            client.trainer.set_client_id(0)
            client.trainer.set_client_id(client_id)
            strategy.client_control_variate = scalar_controls(float(client_id))
        for client_id in (1, 2):
            path = tmp_path / f"models/scaffold_cv_{client_id}.pkl"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(pickle.dumps(scalar_controls(float(client_id))))
        # Assign away/back to import the seeded logical client's canonical state.
        reused.trainer.set_client_id(0)
        for round_id, client_id in enumerate((1, 2, 1, 2), 1):
            reused.client_id = reused._context.client_id = client_id
            reused.configure()
            payload = [
                scalar_controls(1 + round_id / 10),
                scalar_controls(2 + round_id / 10),
            ]
            target = float(client_id)
            expected = local_round(
                dedicated[client_id],
                payload,
                round_id=round_id,
                target=target,
                samples=2,
            )
            actual = local_round(
                reused, payload, round_id=round_id, target=target, samples=2
            )
            torch.testing.assert_close(
                actual[1][0]["theta"], expected[1][0]["theta"], atol=1e-12, rtol=1e-12
            )
            torch.testing.assert_close(
                actual[1][1]["theta"], expected[1][1]["theta"], atol=1e-12, rtol=1e-12
            )
            fresh = create_client(client_id)
            torch.testing.assert_close(
                fresh.trainer.model_update_strategy.client_control_variate["theta"],
                reused.trainer.model_update_strategy.client_control_variate["theta"],
            )


@pytest.mark.parametrize("rate", [0.1, 0.025])
def test_actual_shipped_optimizer_lr_and_next_round_rate(tmp_path, rate):
    config = shipped_config(lr=rate)
    with configure_environment(config, runtime_root=tmp_path):
        assert not hasattr(Config().trainer, "lr")
        client = create_client(1)
        client.trainer.model_update_strategy.client_control_variate = scalar_controls(
            1.0
        )
        received = local_round(
            client,
            [scalar_controls(1.0), scalar_controls(2.0)],
            round_id=1,
            target=1.0,
            samples=2,
        )
        y1 = 1 - rate
        gradient2 = y1 - 1
        expected = y1 - rate * (gradient2 + 1)
        assert client.trainer.model.theta.item() == pytest.approx(expected, abs=1e-12)
        assert received[1][1]["theta"].item() == pytest.approx(
            (0 + gradient2) / 2 - 1, abs=1e-12
        )
        assert client.trainer.model_update_strategy.learning_rate == rate
        # Real registry reads the changed optimizer config in the next round.
        Config.parameters = Config.parameters._replace(
            optimizer=Config.parameters.optimizer._replace(lr=0.05)
        )
        local_round(
            client,
            [scalar_controls(1.0), scalar_controls(2.0)],
            round_id=2,
            target=1.0,
            samples=2,
        )
        assert client.trainer.model_update_strategy.learning_rate == 0.05


@pytest.mark.parametrize("rate", [0.1, 0.025])
def test_config_loads_exact_shipped_toml_before_optimizer_overrides(
    tmp_path, monkeypatch, rate
):
    with isolated_config_state():
        monkeypatch.setenv("config_file", str(CONFIG_PATH))
        monkeypatch.setattr(sys, "argv", ["pytest", "-b", str(tmp_path), "--cpu"])
        config = Config()
        assert config.parameters.optimizer.momentum == 0.9
        assert config.trainer.max_concurrency == 2
        assert not hasattr(config.trainer, "lr")
        Config.parameters = config.parameters._replace(
            optimizer=config.parameters.optimizer._replace(lr=rate, momentum=0.0)
        )
        client = create_client(1)
        strategy = client.trainer.model_update_strategy
        strategy.client_control_variate = scalar_controls(1.0)
        for round_id, actual_rate, initial, server_c in [
            (1, rate, 1.0, 2.0),
            (2, 0.05, 2.0, 3.0),
        ]:
            Config.parameters = Config.parameters._replace(
                optimizer=Config.parameters.optimizer._replace(lr=actual_rate)
            )
            client.trainer.current_round = round_id
            ci = strategy.client_control_variate["theta"].item()
            expected, gradients = initial, []
            for _ in range(2):
                gradients.append(expected)
                expected -= actual_rate * (expected - ci + server_c)
            inbound = ExtractControlVariatesProcessor(1, client.trainer).process(
                [scalar_controls(initial), scalar_controls(server_c)]
            )
            client.algorithm.load_weights(inbound)
            data = TensorDataset(torch.ones(2, 1).double(), torch.zeros(2, 1).double())
            client.trainer.train_model(
                {
                    **config.trainer._asdict(),
                    "run_id": "exact-config",
                    "batch_size": 6,
                    "epochs": 2,
                },
                data,
                [0, 1],
            )
            assert client.trainer.optimizer.param_groups[0]["lr"] == actual_rate
            assert client.trainer.model.theta.item() == pytest.approx(
                expected, abs=1e-12
            )
            assert strategy.client_control_variate["theta"].item() == pytest.approx(
                sum(gradients) / 2, abs=1e-12
            )


def test_equal_count_model_update_reduces_to_uniform_reference(tmp_path):
    with configure_environment(shipped_config(), runtime_root=tmp_path):
        server = make_server()
        server.server_control_variate = scalar_controls(2.0)
        received = []
        for client_id, ci, target in [(1, 1.0, 0.0), (2, 3.0, 2.0)]:
            client = create_client(client_id)
            client.trainer.model_update_strategy.client_control_variate = (
                scalar_controls(ci)
            )
            received.append(
                local_round(
                    client,
                    server.customize_server_payload(server.algorithm.extract_weights()),
                    round_id=1,
                    target=target,
                    samples=2,
                )
            )
        aggregate(server, received)
        assert server.trainer.model.theta.item() == pytest.approx(1.0, abs=1e-12)
        assert server.server_control_variate["theta"].item() == pytest.approx(
            2 / 3, abs=1e-12
        )


def test_unmodified_shipped_momentum_optimizer_composition(tmp_path, caplog):
    config = shipped_config(lr=0.01, momentum=0.9)
    raw_parameters = tomllib.loads(CONFIG_PATH.read_text())["parameters"]["optimizer"]
    assert config["parameters"]["optimizer"] == raw_parameters
    with configure_environment(config, runtime_root=tmp_path):
        client = create_client(1)
        client.trainer.model_update_strategy.client_control_variate = scalar_controls(
            1.0
        )
        reference = ScalarModel()
        optimizer = torch.optim.SGD(reference.parameters(), **raw_parameters)
        for _ in range(2):
            optimizer.zero_grad()
            quadratic_loss(
                reference(torch.ones(2, 1).double()), torch.zeros(2, 1).double()
            ).backward()
            optimizer.step()
            with torch.no_grad():
                reference.theta.sub_(0.01 * (2 - 1))
        actual = local_round(
            client,
            [scalar_controls(1.0), scalar_controls(2.0)],
            round_id=1,
            target=0.0,
            samples=2,
        )
        torch.testing.assert_close(actual[1][0]["theta"], reference.theta)
        expected_ci = 1 - 2 + (1 - reference.theta.item()) / (0.01 * 2)
        assert client.trainer.model_update_strategy.client_control_variate[
            "theta"
        ].item() == pytest.approx(expected_ci, abs=1e-12)
        assert caplog.text.count("additive-control optimizer extension") == 1


def test_reused_processors_rebind_and_missing_inbound_c_clears_previous_round(tmp_path):
    with configure_environment(shipped_config(), runtime_root=tmp_path):
        first, second = create_client(1), create_client(2)
        callback = ScaffoldCallback()
        callback.on_inbound_received(first, first.inbound_processor)
        callback.on_outbound_ready(first, None, first.outbound_processor)
        callback.on_inbound_received(second, first.inbound_processor)
        callback.on_outbound_ready(second, None, first.outbound_processor)
        inbound = next(
            p
            for p in first.inbound_processor.processors
            if isinstance(p, ExtractControlVariatesProcessor)
        )
        outbound = next(
            p
            for p in first.outbound_processor.processors
            if isinstance(p, SendControlVariateProcessor)
        )
        assert inbound.trainer is outbound.trainer is second.trainer
        assert inbound.client_id == outbound.client_id == 2
        inbound.process([scalar_controls(1.0), scalar_controls(3.0)])
        assert (
            second.trainer.context.state["server_control_variate"]["theta"].item()
            == 3.0
        )
        for bad in (
            scalar_controls(1.0),
            [scalar_controls(1.0)],
            [scalar_controls(1.0), None],
        ):
            with pytest.raises(ValueError, match="SCAFFOLD"):
                inbound.process(bad)
            assert second.trainer.context.state.get("server_control_variate") is None
            assert second.trainer.additional_data is None
            with pytest.raises(RuntimeError, match="successful"):
                outbound.process(scalar_controls(1.0))


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        {},
        {"theta": torch.zeros(2).double()},
        {"theta": torch.tensor([float("inf")]).double()},
        {"theta": torch.zeros(1).double(), "unknown": torch.zeros(1)},
    ],
)
def test_all_server_deltas_validated_before_model_or_control_commit(
    tmp_path, malformed
):
    with configure_environment(shipped_config(), runtime_root=tmp_path):
        server = make_server()
        server.server_control_variate = scalar_controls(2.0)
        before = server.algorithm.extract_weights()
        server.updates = [
            SimpleNamespace(
                report=SimpleNamespace(num_samples=2),
                payload=[scalar_controls(7.0), scalar_controls(3.0)],
            ),
            SimpleNamespace(
                report=SimpleNamespace(num_samples=6),
                payload=[scalar_controls(9.0), malformed],
            ),
        ]
        with pytest.raises(ValueError, match="SCAFFOLD"):
            asyncio.run(server._process_reports())
        torch.testing.assert_close(
            server.algorithm.extract_weights()["theta"], before["theta"]
        )
        assert server.server_control_variate["theta"].item() == 2.0
        assert server._pending_control_variate is None


def test_server_buffer_payloads_keep_inherited_aggregation_policy(tmp_path):
    with configure_environment(shipped_config(), runtime_root=tmp_path):
        server = make_server(MixedParameters)
        payload = server.customize_server_payload(server.algorithm.extract_weights())
        assert "integer" in payload[0] and "integer" not in payload[1]
        assert "floating" not in payload[1] and "frozen" not in payload[1]
        received = []
        for count, floating, integer in [(2, 3.0, 1), (6, 7.0, 5)]:
            weights = copy.deepcopy(payload[0])
            weights["floating"].fill_(floating)
            weights["integer"].fill_(integer)
            received.append((SimpleNamespace(num_samples=count), [weights, payload[1]]))
        aggregate(server, received)
        assert server.trainer.model.floating.item() == pytest.approx(6.0)
        assert server.trainer.model.integer.item() == 4
        assert "integer" not in server.server_control_variate
