"""Pytest fixtures shared across test modules."""

import importlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from plato.config import Config
from tests.integration.utils import (
    configure_environment,
    isolated_config_state,
)
from tests.test_utils.fakes import (
    FakeDatasource,
    FakeModel,
    IdentityLifecycleStrategy,
    InMemoryReportingStrategy,
    NoOpCommunicationStrategy,
    RecordingPayloadStrategy,
    StaticTrainingStrategy,
    WeightedAverageAggregation,
)

_DP_MODULE = "tests/trainers/test_dp_data_loader_strategy.py"
_DP_REASON = "base profile optional dp: opacus not installed"
_REQUIRED = {
    _DP_MODULE + "::test_dp_strategy_handles_plato_sampler_get",
    _DP_MODULE + "::test_dp_strategy_handles_torch_sampler_directly",
    "tests/mpc/test_mpc.py::test_round_info_store_local",
    "tests/mpc/test_mpc.py::test_additive_strategy_end_to_end",
    "tests/mpc/test_mpc.py::test_shamir_strategy_end_to_end",
    "tests/integration/test_smoke_configs.py::test_mpc_training_smoke",
}
_STARTUP = "tests/integration/test_event_loop_startup.py::"
_RUNTIME = {
    _STARTUP + "test_real_two_client_cpu_socket_round",
    _STARTUP + "test_post_launch_failure_keeps_primary_error_and_contains_children",
    _STARTUP + "test_stalled_real_client_is_contained_without_round_success",
}
_CORE_RUNTIME = "runtime and not retained_model_search"
_RETAINED_MODULE = "tests/integration/test_retained_model_search.py"
_RETAINED = {
    f"{_RETAINED_MODULE}::{name}"
    for name in (
        "test_retained_width_round[anycostfl]",
        "test_retained_width_round[fedrolex]",
        "test_retained_width_round[heterofl]",
        "test_retained_activated_budget[anycostfl]",
        "test_retained_activated_budget[fedrolex]",
        "test_retained_activated_budget[heterofl]",
        "test_sysheterofl_subnet_round",
        "test_dlg_model_update[lenet]",
        "test_dlg_model_update[resnet_18]",
        "test_local_retired_selection[anycostfl]",
        "test_local_retired_selection[fedrolex]",
        "test_local_retired_selection[heterofl]",
        "test_local_retired_selection[dlg]",
        "test_retained_entrypoint_import[anycostfl-root]",
        "test_retained_entrypoint_import[anycostfl-example]",
        "test_retained_entrypoint_import[fedrolex-root]",
        "test_retained_entrypoint_import[fedrolex-example]",
        "test_retained_entrypoint_import[heterofl-root]",
        "test_retained_entrypoint_import[heterofl-example]",
    )
}
# Fixed delivered 1B cases; update only with the accepted startup test handoff.
_STARTUP_CASES = {
    "test_fresh_entrypoint_executes_scheduled_work": 7,
    "test_active_failure_reaches_synchronous_caller": 4,
    "test_preconstructed_borrowed_loop_resources_survive": 4,
    "test_owned_client_finalizes_tasks_generators_and_executor": 2,
    "test_repeated_owned_client_entrypoints": 1,
    "test_explicitly_absent_loop_on_server_and_edge": 4,
    "test_closed_current_client_loop_is_replaced": 1,
    "test_running_loop_rejected_before_entrypoint_resources": 1,
    "test_delegating_server_consumes_same_borrowed_loop": 1,
    "test_owned_server_fallback_finalizes_its_resources": 2,
    "test_cancellation_already_finalizing_is_not_interrupted": 1,
    "test_runner_cleanup_error_keeps_original_client_exception": 1,
    "test_configure_failure_is_preserved_before_main_coroutine": 1,
    "test_missing_external_edge_client_preserves_value_error": 1,
    "test_client_control_flow_keeps_original_exception": 2,
    "test_active_edge_failure_unwinds_real_aiohttp": 1,
    "test_edge_cancellation_error_keeps_independent_setup_error": 1,
    "test_independent_runtime_error_is_not_mistaken_for_monitor_stop": 1,
    "test_server_error_precedence_and_cooperative_teardown": 5,
    "test_noncooperative_teardown_is_watchdog_containment": 4,
    "test_real_two_client_cpu_socket_round": 1,
    "test_post_launch_failure_keeps_primary_error_and_contains_children": 1,
    "test_stalled_real_client_is_contained_without_round_success": 1,
}


def _full_suite(config) -> bool:
    roots = (config.rootpath.resolve(), (config.rootpath / "tests").resolve())
    return any(
        "::" not in str(arg)
        and (config.invocation_params.dir / str(arg)).resolve() in roots
        for arg in config.args
    )


def _native_prerequisites() -> None:
    """Require the selected Apple Silicon CPU and Metal qualification backend."""
    observed = f"{platform.system()} {platform.machine()}"
    try:
        if observed != "Darwin arm64":
            raise RuntimeError("selected native profile requires Darwin arm64")
        mx = importlib.import_module("mlx.core")
        importlib.import_module("mlx.nn")
        importlib.import_module("mlx.optimizers")
        versions = {
            name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal")
        }
        if versions["mlx"] != versions["mlx-metal"]:
            raise RuntimeError(f"mlx / mlx-metal versions differ: {versions}")
        if not mx.metal.is_available():
            raise RuntimeError("Metal is unavailable")
        for device in (mx.cpu, mx.gpu):
            with mx.stream(device):
                result = mx.sum(mx.array([1.0, 2.0], dtype=mx.float32))
                mx.eval(result)
                mx.synchronize(device)
                scalar = result.item()
                if isinstance(scalar, complex):
                    raise TypeError(f"{device} arithmetic returned complex {scalar}")
                value = float(scalar)
                if not math.isfinite(value) or value != 3.0:
                    raise RuntimeError(f"{device} arithmetic returned {value}")
    except Exception as exc:
        raise pytest.UsageError(
            f"mlx-native prerequisite failed ({observed}): {exc}"
        ) from exc


def _load_native_ledger(path: Path) -> Counter:
    """Read the reviewed full node inventory without deriving it from collection."""
    try:
        ledger = json.loads(path.read_text())
        if not isinstance(ledger, dict) or ledger.get("schema_version") != 1:
            raise ValueError("expected schema_version 1 object")
        cases = ledger.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("cases must be a nonempty list")
        nodes = []
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("each case must be an object")
            node = case.get("nodeid")
            if (
                not isinstance(node, str)
                or not node.startswith("tests/mlx_native/")
                or "::" not in node
                or ".." in Path(node.split("::", 1)[0]).parts
            ):
                raise ValueError(f"invalid native nodeid: {node!r}")
            criteria = case.get("criteria")
            if (
                not isinstance(criteria, list)
                or not criteria
                or any(
                    value not in {f"E{i}" for i in range(1, 8)} for value in criteria
                )
            ):
                raise ValueError(f"invalid E1-E7 criteria for {node}")
            nodes.append(node)
        counts = Counter(nodes)
        duplicate = sorted(node for node, count in counts.items() if count != 1)
        if duplicate:
            raise ValueError(f"duplicate nodeids: {duplicate}")
        return counts
    except (OSError, ValueError, TypeError) as exc:
        raise pytest.UsageError(f"mlx-native ledger invalid: {path}: {exc}") from exc


def pytest_addoption(parser):
    parser.addoption(
        "--test-profile",
        choices=("base", "mandatory", "mlx-native"),
        default=None,
        help=(
            "base permits named optional omissions; unfiltered suites default "
            "mandatory core; mlx-native adds complete native qualification"
        ),
    )


def pytest_configure(config):
    config.pluginmanager.register(_ProfileChecks(config), "plato-profile-checks")


class _ProfileChecks:
    """Enforce declared core/native scopes and their exact case contracts."""

    def __init__(self, config):
        self.config = config
        self.full = _full_suite(config)
        selected = any(
            config.getoption(option) for option in ("keyword", "markexpr", "deselect")
        )
        self.qualify = config.getoption("test_profile") is not None or (
            self.full and not selected
        )
        self.base = config.getoption("test_profile") == "base"
        self.native_boundary = (config.rootpath / "tests/mlx_native").resolve()
        self.native_qualification = config.getoption("test_profile") == "mlx-native"
        explicit_native = any(
            self._native_path(config.invocation_params.dir / str(arg).split("::", 1)[0])
            for arg in config.args
        )
        dotted_native = any(
            str(arg).split("::", 1)[0] == "tests.mlx_native"
            or str(arg).split("::", 1)[0].startswith("tests.mlx_native.")
            for arg in config.args
        )
        self.native_requested = self.native_qualification or explicit_native
        self.native_prerequisite_passed = False
        self.native_expected = Counter()
        self.native_collected = Counter()
        self.session = None
        self.violations = []
        self.passed = set()
        self.allowed_skips = set()
        if config.getoption("pyargs") and (self.native_requested or dotted_native):
            raise pytest.UsageError(self._unsupported_selector())
        if explicit_native and config.getoption("test_profile") in {
            "base",
            "mandatory",
        }:
            raise pytest.UsageError(
                "mlx-native target conflicts with core-only profile; use filesystem "
                "paths without --test-profile for focused checks, or "
                "tests/mlx_native --test-profile=mlx-native"
            )
        markexpr = config.getoption("markexpr") or ""
        self.retained_qualification = (
            config.getoption("test_profile") == "mandatory"
            and markexpr == "retained_model_search"
            and any(
                (config.invocation_params.dir / str(arg).split("::", 1)[0]).resolve()
                == (config.rootpath / _RETAINED_MODULE).resolve()
                for arg in config.args
            )
        )
        if (
            "mlx_native" in re.findall(r"\b\w+\b", markexpr)
            and not self.native_requested
        ):
            raise pytest.UsageError(
                "mlx_native marker requires an explicit native filesystem path "
                "or --test-profile=mlx-native"
            )
        if self.native_qualification and any(
            config.getoption(option)
            for option in ("keyword", "markexpr", "deselect", "ignore", "ignore_glob")
        ):
            raise pytest.UsageError(
                "mlx-native qualification does not permit selection or collection "
                "filters; use an unprofiled native filesystem path for focused checks"
            )
        if (
            self.qualify
            and self.full
            and not self.native_qualification
            and markexpr not in {"", "runtime", "not runtime", _CORE_RUNTIME}
        ):
            raise pytest.UsageError(f"unapproved full-suite partition: {markexpr!r}")
        if self.retained_qualification:
            if (
                self.full
                or len(config.args) != 1
                or "::" in str(config.args[0])
                or config.getoption("pyargs")
            ):
                raise pytest.UsageError(
                    "retained qualification requires the single complete module"
                )
            if any(
                config.getoption(option)
                for option in ("keyword", "deselect", "ignore", "ignore_glob")
            ):
                raise pytest.UsageError(
                    "retained qualification does not permit filters or exclusions"
                )

    def _native_path(self, path) -> bool:
        return Path(path).resolve().is_relative_to(self.native_boundary)

    def _native_node(self, nodeid: str) -> bool:
        return self._native_path(self.config.rootpath / nodeid.split("::", 1)[0])

    @staticmethod
    def _unsupported_selector() -> str:
        return (
            "mlx-native selector unsupported: --pyargs cannot target native tests; "
            "use filesystem paths, e.g. tests/mlx_native --test-profile=mlx-native"
        )

    def pytest_ignore_collect(self, collection_path):
        if not self.native_requested and self._native_path(collection_path):
            return True
        return None

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_collect_file(self, file_path, parent):
        if self._native_path(file_path):
            if self.config.getoption("pyargs"):
                raise pytest.UsageError(self._unsupported_selector())
            if not self.native_requested or not self.native_prerequisite_passed:
                raise pytest.UsageError(
                    "mlx-native preflight required before native file collection; "
                    "use filesystem paths, e.g. tests/mlx_native "
                    "--test-profile=mlx-native"
                )
        return (yield)

    def pytest_sessionstart(self):
        if self.native_requested:
            _native_prerequisites()
            self.native_prerequisite_passed = True
        if self.qualify and not self.base:
            try:
                for name in ("opacus", "kazoo", "gymnasium"):
                    importlib.import_module(name)
            except ImportError as exc:
                raise pytest.UsageError(
                    f"mandatory prerequisite failed: {exc}"
                ) from exc
        if self.native_qualification:
            self.native_expected = _load_native_ledger(
                self.native_boundary / "cases.json"
            )

    def pytest_collection_modifyitems(self, items):
        for item in items:
            if self.retained_qualification and any(
                item.get_closest_marker(name) is None
                for name in ("runtime", "retained_model_search")
            ):
                raise pytest.UsageError(
                    f"retained cases require both markers: {item.nodeid}"
                )
            if self._native_path(item.path):
                if not self.native_requested or not self.native_prerequisite_passed:
                    raise pytest.UsageError("mlx-native collection escaped preflight")
                item.add_marker("mlx_native")
            elif item.get_closest_marker("mlx_native") is not None:
                raise pytest.UsageError(
                    f"mlx_native marker outside native directory: {item.nodeid}"
                )

    def pytest_collection_finish(self, session):
        if self.retained_qualification:
            expected = Counter(_RETAINED)
            collected = Counter(item.nodeid for item in session.items)
            if expected != collected:
                self.violations.append(
                    "retained inventory mismatch: "
                    f"missing {dict(expected - collected)}; "
                    f"extra or duplicate {dict(collected - expected)}"
                )
        self.native_collected = Counter(
            item.nodeid for item in session.items if self._native_path(item.path)
        )
        if self.native_qualification:
            for label, difference in (
                ("missing", self.native_expected - self.native_collected),
                ("extra", self.native_collected - self.native_expected),
            ):
                if difference:
                    self.violations.append(f"native ledger {label}: {dict(difference)}")
            duplicate = sorted(
                node for node, count in self.native_collected.items() if count > 1
            )
            if duplicate:
                self.violations.append(
                    f"native ledger duplicate collection: {duplicate}"
                )

    def _check_skip(self, report):
        if not (self.qualify or self._native_node(report.nodeid)) or not report.skipped:
            return
        reason = report.longrepr[2].removeprefix("Skipped: ")
        allowed = (
            self.base
            and not self._native_node(report.nodeid)
            and report.nodeid == _DP_MODULE
            and reason == _DP_REASON
            and importlib.util.find_spec("opacus") is None
        )
        if allowed:
            self.allowed_skips.add(report.nodeid)
        else:
            self.violations.append(f"unexpected skip: {report.nodeid}: {reason}")

    def pytest_collectreport(self, report):
        self._check_skip(report)

    def pytest_runtest_logreport(self, report):
        if not (self.qualify or self._native_node(report.nodeid)):
            return
        if hasattr(report, "wasxfail"):
            self.violations.append(f"unexpected xfail/xpass: {report.nodeid}")
        else:
            self._check_skip(report)
        if report.when == "call" and report.passed:
            self.passed.add(report.nodeid)

    def pytest_deselected(self, items):
        if not self.qualify:
            return
        partition = self.config.getoption("markexpr")
        for item in items:
            runtime = item.get_closest_marker("runtime") is not None
            retained = item.get_closest_marker("retained_model_search") is not None
            if not (
                (partition == "runtime" and not runtime)
                or (partition == "not runtime" and runtime)
                or (partition == _CORE_RUNTIME and (not runtime or retained))
            ):
                self.violations.append(f"unexpected deselection: {item.nodeid}")

    def pytest_sessionfinish(self, session, exitstatus):
        self.session = session
        if self.retained_qualification and not self.config.getoption("collectonly"):
            missing = _RETAINED - self.passed
            if missing:
                self.violations.append(
                    "retained cases did not pass: " + ", ".join(sorted(missing))
                )
        if self.native_qualification and not self.config.getoption("collectonly"):
            missing = self.native_expected.keys() - self.passed
            if missing:
                self.violations.append(
                    "native ledger cases did not pass: " + ", ".join(sorted(missing))
                )
        if self.qualify and any(
            self.config.getoption(option)
            for option in ("ignore", "ignore_glob", "deselect")
        ):
            self.violations.append("test profiles do not permit collection exclusions")
        if self.qualify and self.full:
            nodes = {item.nodeid for item in session.items}
            partition = self.config.getoption("markexpr")
            runtime = partition in {"runtime", _CORE_RUNTIME}
            required = _RUNTIME if runtime else _REQUIRED
            if self.base:
                required = required - {
                    node for node in required if node.startswith(_DP_MODULE)
                }
            # Collection must contain the named cases even when skips are permitted.
            missing = required - nodes
            if missing:
                self.violations.append(
                    "required tests omitted: " + ", ".join(sorted(missing))
                )
            if not self.config.getoption("collectonly"):
                missing = required - self.passed - self.allowed_skips
                if missing:
                    self.violations.append(
                        "required tests did not pass: " + ", ".join(sorted(missing))
                    )
            if partition != "not runtime":
                actual = Counter(
                    node.split("[", 1)[0] for node in nodes if node.startswith(_STARTUP)
                )
                expected = {
                    _STARTUP + name: count for name, count in _STARTUP_CASES.items()
                }
                if actual != expected:
                    self.violations.append(
                        "required startup cases omitted or changed: expected 48 "
                        f"cases, collected {sum(actual.values())}; "
                        f"mismatched functions: {sorted(name for name in actual.keys() | expected.keys() if actual[name] != expected.get(name, 0))}"
                    )
        if self.violations and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter):
        success = (
            self.session is not None and self.session.exitstatus == pytest.ExitCode.OK
        )
        if not self.native_requested:
            scope = "core scope (native excluded)"
        elif not self.native_qualification:
            scope = "native focused execution (not complete qualification)"
        elif success and self.config.getoption("collectonly"):
            scope = "native collection-only validation (no execution qualification)"
        elif success:
            scope = "native complete qualification"
        else:
            scope = "native qualification failed"
        terminalreporter.write_line(f"test profile: {scope}")
        if self.violations:
            terminalreporter.section("test profile violations", red=True)
            for violation in sorted(set(self.violations)):
                terminalreporter.write_line(violation, red=True)


@pytest.fixture(autouse=True)
def isolate_test_state():
    """Restore Config and CPU RNGs without creating loops or accelerator state."""
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    previous_path = sys.path[:]
    try:
        with isolated_config_state():
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        sys.path[:] = previous_path


@pytest.fixture
def temp_config(tmp_path):
    """Provide an isolated configuration for tests relying on Config."""

    config_data = {
        "clients": {
            "type": "simple",
            "total_clients": 2,
            "per_round": 2,
            "do_test": False,
        },
        "server": {"address": "127.0.0.1", "port": 8000},
        "data": {
            "datasource": "toy",
            "partition_size": 4,
            "sampler": "iid",
            "random_seed": 1,
        },
        "trainer": {
            "type": "basic",
            "rounds": 1,
            "epochs": 1,
            "batch_size": 2,
            "optimizer": "SGD",
            "model_name": "toy_model",
        },
        "algorithm": {"type": "fedavg"},
        "parameters": {"optimizer": {"lr": 0.1, "momentum": 0.0, "weight_decay": 0.0}},
    }

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    with configure_environment(config_data, runtime_root=runtime_root) as config:
        Config.args.id = 1
        yield config


@pytest.fixture
def fake_model_cls():
    """Return the lightweight fake model class for composing components."""
    return FakeModel


@pytest.fixture
def fake_datasource_cls():
    """Return the fake datasource class to create deterministic datasets."""
    return FakeDatasource


@pytest.fixture
def fake_training_strategy():
    """Instantiate a training strategy that skips optimisation."""
    return StaticTrainingStrategy()


@pytest.fixture
def fake_lifecycle_strategy(fake_datasource_cls):
    """Lifecycle strategy that injects fake datasource/trainer components."""
    return IdentityLifecycleStrategy(datasource_factory=fake_datasource_cls)


@pytest.fixture
def fake_reporting_strategy():
    """Reporting strategy storing the most recent report in memory."""
    return InMemoryReportingStrategy()


@pytest.fixture
def fake_communication_strategy():
    """Communication strategy that records outbound artefacts."""
    return NoOpCommunicationStrategy()


@pytest.fixture
def recording_payload_strategy():
    """Payload strategy that records lifecycle events for assertions."""
    return RecordingPayloadStrategy()


@pytest.fixture
def fake_aggregation_strategy():
    """Aggregation strategy performing a simple weighted average."""
    return WeightedAverageAggregation()
