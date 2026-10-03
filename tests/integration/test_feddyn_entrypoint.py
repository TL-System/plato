"""Actual shipped main/run/--resume, spawn, sockets and CPU RNG continuation."""

import copy
import random
from fractions import Fraction

import pytest
import torch

from plato.trainers.strategies.algorithms.feddyn_strategy import same_state
from tests.integration.feddyn_entrypoint_harness import run
from tests.integration.test_feddyn_round_flow import configuration, rational_round

pytestmark = pytest.mark.runtime


def options(root, mode, rounds, kind="scalar"):
    config = configuration(mode, tuple(range(1, 11)), spawn=True, population=10)
    config["clients"].update(per_round=2, comm_simulation=False)
    config["server"].update(
        checkpoint_path=str(root / "checkpoints"),
        model_path=str(root / "models"),
        periodic_interval=0.1,
    )
    config["data"].update(download=False, sampler="feddyn_fixture", reload_data=True)
    config["trainer"].update(rounds=rounds, batch_size=10, max_concurrency=2)
    config["general"] = dict(base_path=str(root))
    if kind == "lenet":
        config["clients"].update(total_clients=2, per_round=1)
        config["trainer"].update(epochs=1, model_name="lenet5", max_concurrency=1)
    return config


def load(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def assert_equations(directory, rounds, *, mode, initial_x=Fraction(2), histories=None):
    histories = histories or [Fraction(0)] * 10
    x = initial_x
    selector = random.Random(17)
    choices = [selector.sample(list(range(1, 11)), 2) for _ in range(4)]
    for round_id in rounds:
        before = load(directory / f"received-{round_id}.pth")
        committed = load(directory / f"committed-{round_id}.pth")
        ids = choices[round_id - 1]
        assert before["selected"] == ids
        x, histories, endpoints = rational_round(
            x,
            histories,
            ids,
            list(map(Fraction, range(1, 11))),
            list(range(1, 11)),
            mode,
        )
        for payload in before["payloads"]:
            i = int(payload[1]["client_id"])
            assert payload[0]["theta"].item() == pytest.approx(
                float(endpoints[i]), abs=1e-12
            )
        assert committed["model"]["theta"].item() == pytest.approx(float(x), abs=1e-12)
        for i, h in enumerate(histories, 1):
            assert committed["histories"][i]["theta"].item() == pytest.approx(
                float(h), abs=1e-12
            )
    return x, histories


@pytest.mark.parametrize("mode", ["uniform", "sample"])
def test_actual_shipped_run_resume_matches_four_round_control_and_rng(tmp_path, mode):
    control_root, split_root = tmp_path / "control", tmp_path / "split"
    control = control_root / "run"
    run(control, options(control_root, mode, 4))
    assert_equations(control, range(1, 5), mode=mode)
    first = split_root / "first"
    run(first, options(split_root, mode, 2))
    x, h = assert_equations(first, range(1, 3), mode=mode)
    saved = load(first / "committed-2.pth")
    assert saved["rng"]["python"] != saved["rng"]["selection"]
    resumed = split_root / "resumed"
    run(resumed, options(split_root, mode, 4), resume=True)
    start = load(resumed / "start.pth")
    assert start["pending"] is None and start["round"] == 2
    assert same_state(start["rng"], saved["rng"])
    assert_equations(resumed, (3, 4), mode=mode, initial_x=x, histories=h)
    for r in (3, 4):
        uninterrupted, restarted = (
            load(control / f"committed-{r}.pth"),
            load(resumed / f"committed-{r}.pth"),
        )
        for key in ("model", "histories", "counts", "committed_round", "rng"):
            assert same_state(uninterrupted[key], restarted[key]), key
    fresh_start = load(control / "start.pth")
    assert fresh_start["round"] == 0 and fresh_start["pending"] is None
    assert fresh_start["rng"]["selection"] == random.Random(17).getstate()


def test_shipped_mnist_shaped_lenet_spawn_and_real_socket_cloud(tmp_path):
    directory = tmp_path / "lenet-run"
    result = run(directory, options(tmp_path, "uniform", 3, "lenet"), kind="lenet")
    assert sum(e["event"] == "custom_server" for e in result["events"]) == 1
    histories = None
    changed = False
    for r in range(1, 4):
        received, committed = (
            load(directory / f"received-{r}.pth"),
            load(directory / f"committed-{r}.pth"),
        )
        histories = (
            copy.deepcopy(received["before"]) if histories is None else histories
        )
        for payload in received["payloads"]:
            i = int(payload[1]["client_id"])
            for k, value in payload[0].items():
                displacement = value - received["baseline"][k]
                changed |= bool(torch.any(displacement != 0))
                histories[i][k] += displacement
        for k, actual in committed["model"].items():
            endpoint_mean = sum(p[0][k] for p in received["payloads"]) / len(
                received["payloads"]
            )
            expected = endpoint_mean + sum(h[k] for h in histories.values()) / 2
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        for i, history in histories.items():
            for name, expected in history.items():
                torch.testing.assert_close(
                    committed["histories"][i][name], expected, atol=1e-6, rtol=1e-6
                )
    assert changed
