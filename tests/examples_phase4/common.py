"""Small, worker-isolated Phase4 configuration and execution helpers.

Workers read PLATO_PHASE4_SPEC and use its authored fields with configured_case.
Runtime paths/ports are supplied separately in the spec file's runtime object.
Spawners synchronously emit child_started before exiting or reparenting. Raw
unacknowledged hints fail qualification; an unproven orphan cannot be signaled.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from plato.config import Config

REPO = Path(__file__).resolve().parents[2]


class _ModuleSource(TypedDict):
    file: str
    loader_origin: str | None
    sha256: str


class _Snapshot(TypedDict):
    files: dict[str, str]
    modules: dict[str, _ModuleSource]
    identity_errors: list[str]
    executable: str
    version: str


def _strings(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(p, str) for p in value):
        raise ValueError("expected a list of strings")
    return cast(list[str], value)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError("expected a mapping with string keys")
    return cast(Mapping[str, object], value)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _path(repo: Path, relative: str) -> Path:
    path = (repo / relative).resolve()
    if not path.is_relative_to(repo.resolve()):
        raise ValueError(f"path escapes checkout: {relative}")
    return path


def source_snapshot(repo: Path, explicit_paths: Sequence[str]) -> dict[str, object]:
    """Hash exercised source and record checkout and interpreter identities."""
    repo = repo.resolve()
    paths = {_path(repo, p) for p in explicit_paths}
    modules, errors = {}, []
    for name, module in tuple(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        path = Path(origin).resolve()
        relevant = name in {"__main__", "__mp_main__"} or path.is_relative_to(repo)
        if name == "plato" or name.startswith(("plato.", "examples.")):
            if not path.is_relative_to(repo):
                errors.append(f"wrong source origin: {name}: {path}")
        if relevant and path.is_file():
            if path.is_relative_to(repo) and (
                name in {"__main__", "__mp_main__"}
                or path.relative_to(repo).parts[0] in {"plato", "examples"}
            ):
                paths.add(path)
            spec = getattr(module, "__spec__", None)
            modules[name] = {
                "file": str(path),
                "loader_origin": getattr(spec, "origin", None),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    lock = repo / "uv.lock"
    return {
        "repo": str(repo),
        "head": head.stdout.strip(),
        "files": hashes,
        "modules": modules,
        "identity_errors": errors,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "version": sys.version,
        "pythonpath": sys.path[:],
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()
        if lock.exists()
        else None,
        "distributions": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
    }


def _identity(pid: object, created: object) -> tuple[int, float]:
    if type(pid) is not int or not 0 < pid < 2**31 or type(created) not in (int, float):
        raise ValueError("invalid process identity")
    try:
        timestamp = float(cast(int | float, created))
    except OverflowError as exc:
        raise ValueError("process creation timestamp overflow") from exc
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise ValueError("invalid process creation timestamp")
    return pid, timestamp


def _receive(connection: socket.socket, deadline: float) -> Mapping[str, object]:
    data = bytearray()
    while b"\n" not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("child registration deadline expired")
        connection.settimeout(remaining)
        chunk = connection.recv(min(4096, 8193 - len(data)))
        if not chunk:
            raise ConnectionError("child registration closed before acknowledgment")
        data.extend(chunk)
        if len(data) > 8192:
            raise ValueError("child registration message exceeds 8192 bytes")
    line, _, extra = data.partition(b"\n")
    if extra:
        raise ValueError("unexpected trailing registration data")
    try:
        return _mapping(json.loads(line.decode("utf-8")))
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("malformed child registration JSON") from exc


def _send(connection: socket.socket, message: object, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("child registration deadline expired")
    connection.settimeout(remaining)
    payload = json.dumps(message, allow_nan=False).encode() + b"\n"
    if len(payload) > 8192:
        raise ValueError("child registration message exceeds 8192 bytes")
    connection.sendall(payload)


def _local_descendant(pid: int, created: float) -> bool:
    import psutil

    try:
        child = psutil.Process(pid)
        reporter = psutil.Process()
        return child.create_time() == created and any(
            parent.pid == reporter.pid
            and parent.create_time() == reporter.create_time()
            for parent in child.parents()
        )
    except psutil.Error:
        return False


def _local_cleanup(pid: int, created: float) -> dict[str, object]:
    import psutil

    forced, errors = [], []
    live = False
    for action, sig in (("TERM", signal.SIGTERM), ("KILL", signal.SIGKILL)):
        try:
            child = psutil.Process(pid)
            live = (
                child.create_time() == created
                and child.status() != psutil.STATUS_ZOMBIE
            )
            if not live:
                break
            forced.append(action)
            child.send_signal(sig)
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                    live = False
                    break
                time.sleep(0.02)
        except psutil.NoSuchProcess:
            live = False
            break
        except psutil.Error as exc:
            errors.append(f"{action}: {type(exc).__name__}: {exc}")
    return {"forced": forced, "survivors": [pid] if live else [], "errors": errors}


def _write_event(
    directory: Path, event: str, fields: Mapping[str, object]
) -> dict[str, object]:
    import psutil

    directory.mkdir(parents=True, exist_ok=True)
    record = {
        **fields,
        "case_id": fields.get("case_id", os.environ.get("PLATO_PHASE4_CASE")),
        "pid": os.getpid(),
        "created": psutil.Process().create_time(),
        "monotonic": time.monotonic(),
        "event": event,
    }
    with (directory / f"events-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
    return record


def emit(directory: Path, event: str, **fields: object) -> None:
    """Record an event; child registration waits for independently verified ownership.

    The spawning process must register immediately and cannot exit before this
    call returns. A rejected/expired registration is fatal worker evidence.
    """
    reserved = ("pid", "created", "monotonic", "event", "registration_id")
    if event != "child_started":
        if any(k in fields for k in reserved):
            raise ValueError("reserved event identity field")
        _write_event(directory, event, fields)
        return
    import psutil

    request_id = secrets.token_hex(16)
    local_identity = None
    try:
        # Caller metadata may not even be serializable. Cleanup authority comes
        # from current OS lineage and creation time before validating that data.
        candidate_pid = fields.get("child_pid")
        if type(candidate_pid) is int and 0 < candidate_pid < 2**31:
            try:
                actual_created = psutil.Process(candidate_pid).create_time()
                if _local_descendant(candidate_pid, actual_created):
                    local_identity = (candidate_pid, actual_created)
            except psutil.Error:
                pass
        if any(k in fields for k in reserved):
            raise ValueError("reserved event identity field")
        pid, created = _identity(fields.get("child_pid"), fields.get("child_created"))
        if pid == os.getpid():
            raise ValueError("a process cannot register itself as its child")
        if local_identity is not None and local_identity != (pid, created):
            raise ValueError("child identity differs from independently proven child")
        record = _write_event(
            directory, event, {**fields, "registration_id": request_id}
        )
        endpoint = _mapping(json.loads(os.environ["PLATO_PHASE4_REGISTRATION"]))
        port, token = endpoint.get("port"), endpoint.get("token")
        if type(port) is not int or not 0 < port < 65536 or not isinstance(token, str):
            raise ValueError("invalid child registration endpoint")
        request = {
            "token": token,
            "case_id": record["case_id"],
            "registration_id": request_id,
            "reporter_pid": record["pid"],
            "reporter_created": record["created"],
            "child_pid": pid,
            "child_created": created,
        }
        deadline = time.monotonic() + 5
        with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
            _send(connection, request, deadline)
            response = _receive(connection, deadline)
        if any(
            response.get(k) != v for k, v in request.items() if k != "token"
        ) or response.get("disposition") not in {"registered", "already_gone"}:
            raise RuntimeError("child registration was rejected or mismatched")
    except (
        KeyError,
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
        RecursionError,
        psutil.Error,
    ) as exc:
        cleanup = (
            _local_cleanup(*local_identity)
            if local_identity
            else {"forced": [], "survivors": []}
        )
        _write_event(
            directory,
            "failure",
            {
                # Never retry serialization of caller-provided failure fields.
                "case_id": os.environ.get("PLATO_PHASE4_CASE"),
                "type": "ChildRegistrationError",
                "message": (
                    f"child registration failed: {type(exc).__name__}: {str(exc)[:512]}"
                ),
                "registration_id": request_id,
                "local_child_owned": local_identity is not None,
                "local_child_identity": {
                    "pid": local_identity[0],
                    "created": local_identity[1],
                }
                if local_identity
                else None,
                "local_cleanup": cleanup,
            },
        )
        raise RuntimeError("child registration failed") from exc


def _leaves(value: Mapping, prefix: str = "") -> Iterator[tuple[str, object]]:
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            if not item:
                raise ValueError(f"empty overlay table: {path}")
            yield from _leaves(item, path)
        else:
            yield path, item


def _replace(data: dict, path: str, value: object) -> dict:
    keys = path.split(".")
    target = data
    for key in keys[:-1]:
        target = target.setdefault(key, {})
        if not isinstance(target, dict):
            raise ValueError(f"overlay parent is not a table: {path}")
    present = keys[-1] in target
    before = copy.deepcopy(target.get(keys[-1]))
    target[keys[-1]] = copy.deepcopy(value)
    return {"path": path, "before_present": present, "before": before, "after": value}


@contextmanager
def configured_case(spec: Mapping[str, object], directory: Path) -> Iterator[Config]:
    """Load the real include graph, apply exact overlays, and isolate Config."""
    import numpy as np
    import torch

    from plato.config import Config, TomlConfigLoader
    from plato.utils import toml_writer
    from tests.integration.utils import isolated_config_state

    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = _path(REPO, str(spec["config"]))
    entrypoint = _path(REPO, str(spec["entrypoint"]))
    visited, stack = [], []

    class ObservedLoader(TomlConfigLoader):
        def _load_file(self, filename, seen):
            visited.append(
                {
                    "path": str(filename),
                    "parent": str(stack[-1]) if stack else None,
                    "sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
                }
            )
            stack.append(filename)
            try:
                return super()._load_file(filename, seen)
            finally:
                stack.pop()

    previous_env, previous_argv = os.environ.copy(), sys.argv
    previous_cwd, previous_path = Path.cwd(), sys.path[:]
    rngs = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    explicit = list(
        dict.fromkeys(
            [
                str(spec["config"]),
                str(spec["entrypoint"]),
                *_strings(spec.get("source_paths", [])),
                "tests/examples_phase4/common.py",
                "tests/integration/utils.py",
            ]
        )
    )
    initial = cast(_Snapshot, source_snapshot(REPO, explicit))
    try:
        if sys.version_info[:2] != (3, 13):
            raise RuntimeError("Phase4 workers require Python 3.13")
        for name in _strings(spec.get("required_distributions", [])):
            importlib.metadata.version(name)
        resolved = ObservedLoader(source).load()
        effective = copy.deepcopy(resolved)
        allowed = spec.get("allowed_overlay_paths", {})
        changes = []
        for path, value in _leaves(_mapping(spec.get("overlays", {}))):
            if not isinstance(allowed, Mapping) or not allowed.get(path):
                raise ValueError(f"unreviewed overlay: {path}")
            changes.append(_replace(effective, path, value))
        runtime_changes = []
        for path, value in {
            "general.base_path": str(directory),
            "data.data_path": "data",
            "server.model_path": "models",
            "server.checkpoint_path": "checkpoints",
            "server.mpc_data_path": "mpc",
            "results.result_path": "results",
        }.items():
            runtime_changes.append(_replace(effective, path, value))
        runtime = _mapping(spec.get("runtime", {}))
        for path, value in _mapping(runtime.get("ports", {})).items():
            runtime_changes.append(_replace(effective, path, value))
        config_path = directory / "effective.toml"
        toml_writer.dump(effective, config_path)
        provenance = {
            "source": str(source),
            "includes": visited,
            "resolved": resolved,
            "effective": effective,
            "authored_overlay_diff": changes,
            "runtime_overlay_diff": runtime_changes,
            "cpu": True,
            "argv": [
                "phase4-worker",
                "--config",
                str(config_path),
                "--base",
                str(directory),
                "--cpu",
            ],
            "seed": spec["seed"],
        }
        (directory / "configuration.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        os.environ["config_file"] = str(config_path)
        sys.argv = provenance["argv"]
        os.chdir(_path(REPO, str(spec.get("cwd", "."))))
        sys.path[:] = [
            *([str(entrypoint.parent)] if entrypoint.parent != REPO else []),
            str(REPO),
            *[p for p in previous_path if p not in {str(REPO), str(entrypoint.parent)}],
        ]
        seed = spec["seed"]
        if type(seed) is not int:
            raise ValueError("case seed must be an integer")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        with isolated_config_state():
            config = Config()
            Config.args.id = 0
            if Config.config._asdict() != effective or Config.device() != "cpu":
                raise AssertionError("effective Config or CPU controls differ")
            emit(
                directory,
                "configured",
                case_id=spec["case_id"],
                configuration=provenance,
            )
            yield config
    except BaseException as exc:
        emit(
            directory,
            "failure",
            case_id=spec["case_id"],
            type=type(exc).__name__,
            message=str(exc),
        )
        raise
    finally:
        try:
            explicit.extend(str(Path(v["path"]).relative_to(REPO)) for v in visited)
            final = cast(_Snapshot, source_snapshot(REPO, explicit))
            errors = final["identity_errors"]
            for path, digest in initial["files"].items():
                if final["files"].get(path) != digest:
                    errors.append(f"source changed: {path}")
            for item in visited:
                if final["files"].get(item["path"]) != item["sha256"]:
                    errors.append(f"included source changed: {item['path']}")
            for name, module in tuple(sys.modules.items()):
                origin = getattr(module, "__file__", None)
                if (
                    name == entrypoint.stem
                    and origin
                    and Path(origin).resolve() != entrypoint
                ):
                    errors.append(f"wrong entrypoint origin: {name}: {origin}")
            emit(directory, "provenance", case_id=spec["case_id"], snapshot=final)
            if errors:
                raise AssertionError("; ".join(errors))
        finally:
            os.environ.clear()
            os.environ.update(previous_env)
            sys.argv, sys.path[:] = previous_argv, previous_path
            os.chdir(previous_cwd)
            random.setstate(rngs[0])
            np.random.set_state(rngs[1])
            torch.set_rng_state(rngs[2])
            torch.set_num_threads(threads)


def run_case(
    directory: Path,
    *,
    worker: str,
    spec: Mapping[str, object],
    interpreter: str | None = None,
) -> dict[str, object]:
    """Run one bounded CPU worker and retain evidence before rejecting failures."""
    import psutil

    authored = copy.deepcopy(dict(spec))
    if "runtime" in authored:
        raise ValueError("runtime expansion is owned by run_case")
    strict = os.environ.get("PLATO_PHASE4_STRICT_TASK")
    ledger_hash = None
    if strict:
        ledger_path = Path(os.environ["PLATO_PHASE4_LEDGER"])
        ledger_hash = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
        ledger = json.loads(ledger_path.read_text())
        row = next(t for t in ledger["tasks"] if t["task_id"] == strict)
        case = next((c for c in row["cases"] if c["case_id"] == spec["case_id"]), None)
        if (
            row["state"] != "frozen"
            or worker != row["worker_module"]
            or case is None
            or _digest(authored) != case["authored_spec_sha256"]
        ):
            raise AssertionError("worker or authored spec differs from frozen ledger")
    timeout_value = spec.get("timeout_seconds", 60)
    if not isinstance(timeout_value, (int, float)):
        raise ValueError("worker timeout must be numeric")
    timeout = float(timeout_value)
    if not 0 < timeout <= 180:
        raise ValueError("worker timeout must be in (0, 180]")
    directory = directory.resolve()
    if directory.is_relative_to(REPO):
        raise ValueError("runtime directory must be outside checkout")
    directory.mkdir(parents=True, exist_ok=False)
    ports = {}
    if spec.get("transport") == "socket":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            ports["server.port"] = sock.getsockname()[1]
    runtime = {"directory": str(directory), "ports": ports}
    effective_spec = {**authored, "runtime": runtime}
    spec_path = directory / "spec.json"
    spec_path.write_text(json.dumps(effective_spec, indent=2) + "\n")
    explicit = list(
        dict.fromkeys(
            [
                str(spec["config"]),
                str(spec["entrypoint"]),
                *_strings(spec.get("source_paths", [])),
                "tests/examples_phase4/common.py",
                "tests/integration/utils.py",
                worker.replace(".", "/") + ".py",
            ]
        )
    )
    initial = cast(_Snapshot, source_snapshot(REPO, explicit))
    cwd = _path(REPO, str(spec.get("cwd", ".")))
    path_order = [str(REPO)]
    entrypoint_parent = _path(REPO, str(spec["entrypoint"])).parent
    if cwd == entrypoint_parent and cwd != REPO:
        path_order.append(str(cwd))
    if spec.get("interpreter_env") and not os.environ.get(str(spec["interpreter_env"])):
        raise ValueError("reviewed worker interpreter environment variable is unset")
    executable = interpreter or (
        os.environ[str(spec["interpreter_env"])]
        if spec.get("interpreter_env")
        else sys.executable
    )
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(path_order),
        "PYTHONHASHSEED": str(spec["seed"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PLATO_PHASE4_SPEC": str(spec_path),
        "PLATO_PHASE4_CASE": str(spec["case_id"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(directory / "hf-cache"),
        **{
            n: "1"
            for n in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    command = [executable, "-B", "-m", worker]
    started = time.monotonic()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(0.05)
    registration_token = secrets.token_hex(32)
    environment["PLATO_PHASE4_REGISTRATION"] = json.dumps(
        {
            "port": listener.getsockname()[1],
            "token": registration_token,
        }
    )
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    identities = {proc.pid: psutil.Process(proc.pid).create_time()}
    finished = threading.Event()
    registration_closed = threading.Event()
    registration_lock = threading.RLock()
    connections: set[socket.socket] = set()
    registrations: dict[str, dict[str, object]] = {}
    registration_errors: list[str] = []
    malformed = []

    def read_events():
        events = []
        try:
            paths = sorted(directory.glob("events-*.jsonl"))
        except OSError as exc:
            malformed.append(f"event listing: {exc}")
            return events
        for path in paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as exc:
                malformed.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            for line in lines:
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or not all(
                        k in event
                        for k in ("pid", "created", "monotonic", "event", "case_id")
                    ):
                        raise ValueError("event lacks required identity fields")
                    if (
                        type(event["pid"]) is not int
                        or event["pid"] <= 0
                        or not isinstance(event["event"], str)
                        or not event["event"]
                        or not isinstance(event["case_id"], str)
                        or any(
                            type(event[k]) not in (int, float)
                            or not math.isfinite(event[k])
                            for k in ("created", "monotonic")
                        )
                        or (
                            event["pid"] == proc.pid
                            and event["created"] != identities[proc.pid]
                        )
                    ):
                        raise ValueError("invalid event identity")
                    if event["event"] == "provenance":
                        snapshot = event.get("snapshot")
                        if not isinstance(snapshot, dict) or (
                            not isinstance(snapshot.get("files"), dict)
                            or not isinstance(snapshot.get("identity_errors"), list)
                            or not isinstance(snapshot.get("executable"), str)
                            or not isinstance(snapshot.get("modules"), dict)
                            or not isinstance(snapshot.get("version"), str)
                            or any(
                                not isinstance(k, str) or not isinstance(v, str)
                                for k, v in snapshot.get("files", {}).items()
                            )
                            or any(
                                not isinstance(v, dict)
                                or not isinstance(v.get("file"), str)
                                for v in snapshot.get("modules", {}).values()
                            )
                        ):
                            raise ValueError("invalid provenance snapshot")
                    events.append(event)
                except (ValueError, TypeError, OverflowError, RecursionError) as exc:
                    malformed.append(f"{path.name}: {exc}")
        return events

    def register_owned(process):
        """Accept OS session/lineage proof, never a worker's ownership claim."""
        try:
            created = process.create_time()
            if process.pid in identities:
                return identities[process.pid] == created
            same_session = os.getsid(process.pid) == proc.pid
            parent_pid = process.ppid()
            parent_owned = False
            if parent_pid in identities:
                parent = psutil.Process(parent_pid)
                parent_owned = parent.create_time() == identities[parent_pid]
            if same_session or parent_owned:
                identities[process.pid] = created
                return True
        except (
            ProcessLookupError,
            PermissionError,
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            pass
        return False

    def refresh_owned():
        for process in psutil.process_iter(["pid", "create_time"]):
            register_owned(process)

    def observe():
        while not finished.wait(0.02):
            refresh_owned()

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()

    def survivors():
        # Cleanup never reads events; corrupt evidence cannot disable containment.
        refresh_owned()
        live = []
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

    def serve_registrations():
        while not registration_closed.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                with registration_lock:
                    if registration_closed.is_set():
                        break
                    connections.add(connection)
                deadline = time.monotonic() + 1
                try:
                    request = _receive(connection, deadline)
                    token = request.get("token")
                    if not isinstance(token, str) or not secrets.compare_digest(
                        token, registration_token
                    ):
                        raise ValueError("child registration authentication failed")
                    request_id = request.get("registration_id")
                    if (
                        request.get("case_id") != spec["case_id"]
                        or not isinstance(request_id, str)
                        or len(request_id) != 32
                    ):
                        raise ValueError("invalid registration request identity")
                    reporter_pid, reporter_created = _identity(
                        request.get("reporter_pid"), request.get("reporter_created")
                    )
                    child_pid, child_created = _identity(
                        request.get("child_pid"), request.get("child_created")
                    )
                    with registration_lock:
                        if registration_closed.is_set() or request_id in registrations:
                            raise ValueError("child registration closed or duplicate")
                        reporter = psutil.Process(reporter_pid)
                        if (
                            reporter.create_time() != reporter_created
                            or reporter.status() == psutil.STATUS_ZOMBIE
                            or not register_owned(reporter)
                            or child_pid == reporter_pid
                        ):
                            raise ValueError("unowned registration reporter")
                        try:
                            child = psutil.Process(child_pid)
                            gone = (
                                child.create_time() != child_created
                                or child.status() == psutil.STATUS_ZOMBIE
                            )
                        except psutil.NoSuchProcess:
                            gone = True
                        if not gone and not register_owned(child):
                            raise ValueError(
                                "child has no independently proven owned lineage/session"
                            )
                        response = {
                            k: request[k]
                            for k in (
                                "case_id",
                                "registration_id",
                                "reporter_pid",
                                "reporter_created",
                                "child_pid",
                                "child_created",
                            )
                        }
                        response["disposition"] = (
                            "already_gone" if gone else "registered"
                        )
                        # Persist proof before acknowledgment releases the spawning parent.
                        registrations[request_id] = response
                    _send(connection, response, deadline)
                except (
                    ValueError,
                    TypeError,
                    OverflowError,
                    OSError,
                    psutil.Error,
                ) as exc:
                    registration_errors.append(f"{type(exc).__name__}: {exc}")
                finally:
                    with registration_lock:
                        connections.discard(connection)

    registrar = threading.Thread(target=serve_registrations, daemon=True)
    registrar.start()

    timed_out, forced, errors = False, [], []
    stdout = stderr = ""
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        if not timed_out:
            deadline = time.monotonic() + 5
            while survivors() and time.monotonic() < deadline:
                time.sleep(0.02)
        for action, sig in (("TERM", signal.SIGTERM), ("KILL", signal.SIGKILL)):
            live = survivors()
            if not live:
                break
            forced.append(action)
            for process in live:
                try:
                    process.send_signal(sig)
                except psutil.NoSuchProcess:
                    pass
            deadline = time.monotonic() + 5
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            while survivors() and time.monotonic() < deadline:
                time.sleep(0.02)
    except BaseException as exc:
        errors.append(f"containment error: {type(exc).__name__}: {exc}")
    finally:
        # Freeze registration before the final containment snapshot. Active partial
        # requests cannot delay shutdown or add ownership after this point.
        with registration_lock:
            registration_closed.set()
            listener.close()
            for connection in list(connections):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        registrar.join(timeout=1.1)
        if registrar.is_alive():
            errors.append("child registration thread did not stop")
        # The fallback also records forced actions; it cannot produce success.
        for process in survivors():
            forced.append(f"finally-KILL:{process.pid}")
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            errors.append("pipes did not close after containment")
        finished.set()
        observer.join(timeout=1)
    events = read_events()
    events.sort(key=lambda e: e["monotonic"])
    unresolved_hints = []
    observed_registrations = set()
    for event in events:
        if event["event"] != "child_started":
            continue
        request_id = event.get("registration_id")
        receipt = registrations.get(request_id) if isinstance(request_id, str) else None
        if isinstance(request_id, str):
            if request_id in observed_registrations:
                errors.append("duplicate child_started registration identity")
            observed_registrations.add(request_id)
        if receipt is None or any(
            receipt.get(k) != event.get(source)
            for k, source in (
                ("case_id", "case_id"),
                ("reporter_pid", "pid"),
                ("reporter_created", "created"),
                ("child_pid", "child_pid"),
                ("child_created", "child_created"),
            )
        ):
            errors.append("child_started lacks matching acknowledged registration")
        try:
            pid, created = _identity(event.get("child_pid"), event.get("child_created"))
            child = psutil.Process(pid)
            if (
                child.create_time() == created
                and child.status() != psutil.STATUS_ZOMBIE
                and identities.get(pid) != created
            ):
                unresolved_hints.append({"pid": pid, "created": created})
        except psutil.NoSuchProcess:
            pass
        except (ValueError, OverflowError, psutil.Error) as exc:
            errors.append(f"invalid or unclassifiable child hint: {exc}")
    if unresolved_hints:
        errors.append(
            "unresolved live child hints; ownership unproven and no signal authorized"
        )
    errors.extend("child registration error: " + e for e in registration_errors)
    live = [p.pid for p in survivors()]
    if proc.returncode != 0 or timed_out or forced or live or malformed:
        errors.append("worker did not exit naturally with valid evidence")
    parent_events = [e for e in events if e["pid"] == proc.pid]
    if (
        not parent_events
        or parent_events[-1]["event"] != "success"
        or sum(e["event"] == "success" for e in parent_events) != 1
    ):
        errors.append("missing unique final parent success event")
    if not parent_events or parent_events[0]["event"] != "worker_started":
        errors.append("missing initial worker_started event")
    if not any(e["event"] == "configured" for e in parent_events):
        errors.append("missing configured event")
    snapshots = [
        cast(_Snapshot, e["snapshot"])
        for e in parent_events
        if e["event"] == "provenance"
    ]
    if not snapshots:
        errors.append("missing source provenance")
    for snapshot in snapshots:
        expected_worker = str(_path(REPO, worker.replace(".", "/") + ".py"))
        if snapshot["modules"].get("__main__", {}).get("file") != expected_worker:
            errors.append("worker module origin mismatch")
        if snapshot["identity_errors"] or any(
            snapshot["files"].get(p) != h for p, h in initial["files"].items()
        ):
            errors.append("source identity mismatch")
        if Path(snapshot["executable"]).absolute() != Path(executable).absolute():
            errors.append("interpreter identity mismatch")
    if any(
        e["case_id"] != spec["case_id"]
        or e["event"] in {"failure", "skip", "xfail", "xpass"}
        for e in events
    ):
        errors.append("failed, skipped, or mismatched case event")
    if any(
        s in stdout + stderr
        for s in (
            "Traceback (most recent call last)",
            "was never awaited",
            "Task was destroyed",
            "never retrieved",
        )
    ):
        errors.append("worker traceback or asynchronous diagnostic")
    result = {
        "case_id": spec["case_id"],
        "command": command,
        "cwd": str(cwd),
        "environment": {
            k: environment[k]
            for k in (
                "PYTHONPATH",
                "PYTHONHASHSEED",
                "PLATO_PHASE4_SPEC",
                "HF_HOME",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "authored_spec": authored,
        "authored_spec_sha256": _digest(authored),
        "ledger_sha256": ledger_hash,
        "runtime": runtime,
        "initial_source": initial,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "forced": forced,
        "survivors": live,
        "processes": identities,
        "events": events,
        "malformed_events": sorted(set(malformed)),
        "child_registrations": list(registrations.values()),
        "registration_errors": registration_errors,
        "unresolved_child_hints": unresolved_hints,
        "errors": errors,
        "elapsed": time.monotonic() - started,
    }
    (directory / "stdout.log").write_text(stdout)
    (directory / "stderr.log").write_text(stderr)
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert not errors, (
        f"Phase4 case failed: {errors}; evidence: {directory / 'result.json'}"
    )
    return result
