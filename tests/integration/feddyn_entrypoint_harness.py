"""Bounded real socket runs with retained, identity-scoped process evidence."""

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from plato.utils import toml_writer
from tests.integration.startup_harness import available_port

REPO = Path(__file__).resolve().parents[2]


def run(directory, config, *, resume=False, kind="scalar", timeout=150):
    directory.mkdir(parents=True, exist_ok=True)
    config["server"]["port"] = available_port()
    toml_writer.dump(config, directory / "config.toml")
    command = [
        sys.executable,
        "-B",
        "-m",
        "tests.integration.feddyn_entrypoint",
        "--cpu",
    ]
    if resume:
        command.append("--resume")
    env = {
        **os.environ,
        "config_file": str(directory / "config.toml"),
        "PYTHONPATH": str(REPO),
        "PLATO_FEDDYN_ROOT": str(directory),
        "PLATO_FEDDYN_KIND": kind,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    proc = subprocess.Popen(
        ["zsh", "-lc", shlex.join(command)],
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
                except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
                    pass

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    timed_out, forced = False, False
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out, forced = True, True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate(timeout=5)
    finally:
        finished.set()
        observer.join(timeout=1)
    events = [
        json.loads(line)
        for path in directory.glob("events-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    for event in events:
        if event["event"] == "child_started":
            identities[event["child_pid"]] = event["created"]
    survivors = []
    deadline = time.monotonic() + 5
    while True:
        survivors = []
        for pid, created in list(identities.items()):
            try:
                p = psutil.Process(pid)
                if p.create_time() == created and p.status() != psutil.STATUS_ZOMBIE:
                    survivors.append(pid)
            except psutil.NoSuchProcess:
                pass
        if not survivors or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    result = dict(
        command=command,
        returncode=proc.returncode,
        timed_out=timed_out,
        forced=forced,
        survivors=survivors,
        processes=identities,
        events=events,
        stdout=stdout,
        stderr=stderr,
    )
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    assert proc.returncode == 0 and not timed_out and not forced and not survivors, (
        stdout + stderr
    )
    # The inherited production close exits the server process with status 0;
    # child entrypoints return normally. Do not replace that runtime lifecycle.
    assert any(e["event"] == "server_will_close" for e in events)
    assert any(e["event"] == "child_returned" for e in events)
    return result
