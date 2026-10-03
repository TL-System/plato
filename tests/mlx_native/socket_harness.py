"""Native socket-round observations with task-owned watchdog containment."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from plato.utils import toml_writer

REPO = Path(__file__).resolve().parents[2]


def available_port() -> int:
    """Choose a loopback port; callers retry only confirmed bind collisions."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_probe(
    directory: Path,
    case: str,
    *,
    timeout: float = 30,
    interpreter: str = sys.executable,
    collision_attempt: int = 0,
    **options,
) -> dict:
    """Drain output, retain process identities, and contain only owned processes."""
    directory.mkdir(parents=True)
    port = available_port()
    config = {
        "clients": {
            "type": "simple",
            "total_clients": 2,
            "per_round": 2,
            "do_test": False,
            "comm_simulation": False,
        },
        "server": {
            "address": "127.0.0.1",
            "port": port,
            "random_seed": 7,
            "do_test": False,
            "disable_clients": True,
        },
        "data": {
            "datasource": "startup_tiny",
            "sampler": "all_inclusive",
            "random_seed": 7,
        },
        "trainer": {
            "type": "mlx",
            "framework": "mlx",
            "rounds": 1,
            "epochs": 1,
            "batch_size": 8,
            "max_concurrency": 2,
            "optimizer": "adam",
            "model_name": "lenet5",
            "model_seed": 17,
            "training_seed": 29,
        },
        "algorithm": {"type": "mlx_fedavg", "framework": "mlx"},
        "parameters": {
            "model": {"framework": "mlx", "num_classes": 10},
            "optimizer": {"learning_rate": 0.001},
        },
        "general": {"base_path": str(directory)},
    }
    if case.startswith("socket"):
        config["server"]["disable_clients"] = False
        config["data"]["sampler"] = "mlx_native_fixture"
        config["clients"]["outbound_processors"] = ["safetensor_encode"]
        config["clients"]["inbound_processors"] = ["safetensor_decode"]
        config["server"]["outbound_processors"] = ["safetensor_encode"]
        config["server"]["inbound_processors"] = ["safetensor_decode"]
        config["trainer"]["batch_size"] = options.get("batch_size", 8)
        config["trainer"]["rounds"] = options.get("rounds", 1)
    if case.startswith("mnist_cached_simulation"):
        config = options["config"]
        config["general"] = {"base_path": str(directory)}
        config["server"]["port"] = port
        config["data"]["data_path"] = options["cached_data"]
    if case.startswith("edge") or case == "central":
        config["algorithm"].update(cross_silo=True, total_silos=1, local_rounds=1)
    toml_writer.dump(config, directory / "config.toml")
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "config_file": str(directory / "config.toml"),
        "PLATO_MLX_ROOT": str(directory),
        "PLATO_MLX_CASE": case,
        "PLATO_MLX_OPTIONS": json.dumps(options),
    }
    command = [interpreter, "-B", "-m", "tests.mlx_native.socket_probe"]
    if case == "mnist_cached_simulation_main":
        command = [
            interpreter,
            "-B",
            str(REPO / "plato.py"),
            "--config",
            str(directory / "config.toml"),
        ]
    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        cwd=directory,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    identities = {proc.pid: psutil.Process(proc.pid).create_time()}
    finished = threading.Event()

    def observe():
        while not finished.wait(0.02):
            for process in psutil.process_iter(["pid", "create_time"]):
                try:
                    if os.getpgid(process.pid) == proc.pid:
                        identities[process.pid] = process.create_time()
                except (ProcessLookupError, psutil.NoSuchProcess, PermissionError):
                    pass

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()

    def survivors():
        live = []
        # Durable Process.start markers also cover very short-lived parent races.
        for path in directory.glob("events-*.jsonl"):
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event["event"] == "child_started":
                    identities[event["child_pid"]] = event["created"]
        for pid, created in list(identities.items()):
            try:
                process = psutil.Process(pid)
                if (
                    process.create_time() == created
                    and process.status() != psutil.STATUS_ZOMBIE
                ):
                    live.append(process)
            except psutil.NoSuchProcess:
                pass
        return live

    timed_out = False
    forced = []
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout = stderr = ""
        if not timed_out:
            natural_deadline = time.monotonic() + 5
            while survivors() and time.monotonic() < natural_deadline:
                time.sleep(0.02)
        if survivors():
            forced.append("TERM")
            deadline = time.monotonic() + 5
            for process in survivors():
                try:
                    process.send_signal(signal.SIGTERM)
                except psutil.NoSuchProcess:
                    pass
            try:
                stdout, stderr = proc.communicate(
                    timeout=max(0.01, deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                pass
            while survivors() and time.monotonic() < deadline:
                time.sleep(0.02)
            if survivors():
                forced.append("KILL")
                deadline = time.monotonic() + 5
                for process in survivors():
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                stdout, stderr = proc.communicate(
                    timeout=max(0.01, deadline - time.monotonic())
                )
                while survivors() and time.monotonic() < deadline:
                    time.sleep(0.02)
    finally:
        finished.set()
        observer.join(timeout=1)
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)
    events = []
    for path in sorted(directory.glob("events-*.jsonl")):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    result = {
        "case": case,
        "options": options,
        "command": command,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "forced": forced,
        "elapsed": time.monotonic() - started,
        "survivors": [p.pid for p in survivors()],
        "processes": identities,
        "events": events,
        "stdout": stdout,
        "stderr": stderr,
        "port": port,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    bind_collision = any(
        event["event"] == "caller_error"
        and event["type"] == "OSError"
        and "address already in use" in event["message"]
        for event in events
    )
    if (
        case.startswith("socket")
        and bind_collision
        and collision_attempt < 2
        and not timed_out
        and not forced
        and not result["survivors"]
    ):
        retry = run_probe(
            directory / "bind-retry",
            case,
            timeout=timeout,
            interpreter=interpreter,
            collision_attempt=collision_attempt + 1,
            **options,
        )
        retry.setdefault("collision_retries", []).insert(
            0, {"result_path": str(directory / "result.json"), "port": port}
        )
        return retry
    return result
