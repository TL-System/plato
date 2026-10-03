"""Isolated qualification of retained model-search paths and local selectors."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = [pytest.mark.runtime, pytest.mark.retained_model_search]
REPOSITORY = Path(__file__).resolve().parents[2]
CONFIGS = {
    "anycostfl": "examples/model_search/anycostfl/example_ResNet.toml",
    "fedrolex": "examples/model_search/fedrolex/example_ResNet.toml",
    "heterofl": "examples/model_search/heterofl/heterofl_resnet18_dynamic.toml",
    "sysheterofl": "examples/model_search/sysheterofl/config_ResNet152.toml",
    "dlg": "examples/gradient_leakage_attacks/fedavg_resnet18_cifar100.toml",
}


def _group_members(group: int) -> list[int]:
    """Observe only the process group created for this worker."""
    rows = subprocess.check_output(
        ["ps", "-axo", "pid=,pgid=,stat="], text=True
    ).splitlines()
    return [
        int(fields[0])
        for row in rows
        if len(fields := row.split()) >= 3
        and int(fields[1]) == group
        and not fields[2].startswith("Z")
    ]


def _run_case(
    directory: Path, family: str, case: str, timeout: int, *, variant: str = ""
) -> dict[str, Any]:
    """Run a main-guarded worker, keeping its config, logs and containment receipt."""
    from plato.utils import toml_writer

    directory.mkdir()
    source = REPOSITORY / CONFIGS[family]
    config = tomllib.loads(source.read_text())
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    overlays: dict[str, Any] = {
        "clients": {
            "total_clients": 2,
            "per_round": 2,
            "comm_simulation": False,
            "compute_comm_time": False,
            "do_test": False,
        },
        "server": {"port": port, "simulate_wall_time": False, "random_seed": 1234},
        "trainer": {
            "rounds": 1,
            "epochs": 1,
            "batch_size": 2,
            "max_concurrency": 2,
        },
        "data": {
            "download": False,
            "sampler": "all_inclusive",
            "testset_sampler": "all_inclusive",
            "testset_size": 2,
        },
        "parameters": {"limitation": {"activated": case == "budget"}},
    }
    if family == "sysheterofl":
        overlays["clients"]["per_round"] = 1
        cast(dict[str, Any], overlays["parameters"]["limitation"]).update(max_loop=1)
        cast(dict[str, Any], overlays["parameters"])["distillation"] = {"iterations": 1}
    if family == "dlg":
        cast(dict[str, Any], overlays["trainer"])["model_type"] = "resnet"
        if variant == "lenet":
            cast(dict[str, Any], overlays["trainer"])["model_name"] = "lenet"
            overlays["data"]["datasource"] = "CIFAR100"

    def update(target: dict[str, Any], patch: dict[str, Any]) -> None:
        for key, value in patch.items():
            if isinstance(value, dict):
                update(target.setdefault(key, {}), value)
            else:
                target[key] = value

    update(config, overlays)
    toml_writer.dump(config, directory / "config.toml")
    (directory / "overlay.json").write_text(json.dumps(overlays, indent=2) + "\n")
    env = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PLATO_RETENTION_ROOT": str(directory),
        "PLATO_RETENTION_FAMILY": family,
        "PLATO_RETENTION_CASE": case,
        "PLATO_RETENTION_VARIANT": variant,
        "config_file": str(directory / "config.toml"),
    }
    command = [
        sys.executable,
        "-B",
        "-m",
        "tests.integration.model_search_retention_worker",
        "-u",
        "-c",
        str(directory / "config.toml"),
        "-b",
        str(directory / "runtime"),
    ]
    cwd = REPOSITORY
    if case == "import" and variant == "example":
        cwd = REPOSITORY / "examples" / "model_search" / family
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    observed = {process.pid}
    finished = threading.Event()

    def observe() -> None:
        while not finished.wait(0.05):
            observed.update(_group_members(process.pid))

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    timed_out = False
    forced = []
    stdout = stderr = ""
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        natural_deadline = time.monotonic() + 5
        while not timed_out and _group_members(process.pid):
            if time.monotonic() >= natural_deadline:
                break
            time.sleep(0.05)
        for name, number in (("TERM", signal.SIGTERM), ("KILL", signal.SIGKILL)):
            if not _group_members(process.pid):
                break
            forced.append(name)
            os.killpg(process.pid, number)
            deadline = time.monotonic() + 5
            while _group_members(process.pid) and time.monotonic() < deadline:
                time.sleep(0.05)
        if process.poll() is None:
            process.kill()
        stdout, stderr = process.communicate(timeout=5)
    finally:
        finished.set()
        observer.join(timeout=1)
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    events = [
        json.loads(line)
        for path in sorted(directory.glob("events-*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    result = {
        "family": family,
        "case": case,
        "variant": variant,
        "command": command,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "forced": forced,
        "survivors": _group_members(process.pid),
        "observed_pids": sorted(observed),
        "elapsed_seconds": time.monotonic() - started,
        "events": events,
        "overlay": overlays,
        "supplied_config_path": CONFIGS[family],
        "supplied_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256(
            (REPOSITORY / "uv.lock").read_bytes()
        ).hexdigest(),
    }
    (directory / "stdout.log").write_text(stdout)
    (directory / "stderr.log").write_text(stderr)
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert not timed_out and not forced and not result["survivors"], result
    assert process.returncode == 0, stderr + stdout
    assert "was never awaited" not in stderr and "Task was destroyed" not in stderr
    assert any(event["event"] == "success" for event in events), result
    return result


@pytest.mark.parametrize("family", ["anycostfl", "fedrolex", "heterofl"])
def test_retained_width_round(tmp_path: Path, family: str) -> None:
    result = _run_case(tmp_path / family, family, "socket", 90)
    rates = {
        event["client_id"]: event["rate"]
        for event in result["events"]
        if event["event"] == "selected_rate"
    }
    assert rates == {1: 0.5, 2: 1.0}
    assert sorted(
        event["samples"]
        for event in result["events"]
        if event["event"] == "client_trained"
    ) == [2, 6]
    assert any(e["event"] == "aggregation_oracle" for e in result["events"])


@pytest.mark.parametrize("family", ["anycostfl", "fedrolex", "heterofl"])
def test_retained_activated_budget(tmp_path: Path, family: str) -> None:
    _run_case(tmp_path / family, family, "budget", 30)


def test_sysheterofl_subnet_round(tmp_path: Path) -> None:
    _run_case(tmp_path / "sysheterofl", "sysheterofl", "subnet", 180)


@pytest.mark.parametrize("model_name", ["lenet", "resnet_18"])
def test_dlg_model_update(tmp_path: Path, model_name: str) -> None:
    _run_case(tmp_path / model_name, "dlg", "update", 90, variant=model_name)


@pytest.mark.parametrize("family", ["anycostfl", "fedrolex", "heterofl", "dlg"])
def test_local_retired_selection(tmp_path: Path, family: str) -> None:
    _run_case(tmp_path / family, family, "selector", 15)


@pytest.mark.parametrize(
    "family,location",
    [
        (family, location)
        for family in ("anycostfl", "fedrolex", "heterofl")
        for location in ("root", "example")
    ],
    ids=[
        f"{family}-{location}"
        for family in ("anycostfl", "fedrolex", "heterofl")
        for location in ("root", "example")
    ],
)
def test_retained_entrypoint_import(tmp_path: Path, family: str, location: str) -> None:
    _run_case(tmp_path / family, family, "import", 15, variant=location)
