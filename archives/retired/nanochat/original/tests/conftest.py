"""Pytest fixtures shared across test modules."""

import importlib
import importlib.util
import random
import subprocess
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
    nanochat_source,
    native_tokenizer_available,
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

_NANOCHAT = "tests/test_nanochat_integration.py::"
_NANOCHAT_TESTS = {
    _NANOCHAT + "test_nanochat_tokenizer_processor_round_trip",
    _NANOCHAT + "test_nanochat_trainer_smoke",
    _NANOCHAT + "test_nanochat_trainer_selects_core_eval_strategy",
}
_DP_MODULE = "tests/trainers/test_dp_data_loader_strategy.py"
_BASE_NANOCHAT_REASON = "base profile omits optional nanochat integration"
_NATIVE_REASON = "optional native nanochat tokenizer: callable rustbpe.Tokenizer absent"
_DP_REASON = "base profile optional dp: opacus not installed"
_REQUIRED = _NANOCHAT_TESTS | {
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
        "::" not in str(arg) and Path(str(arg)).resolve() in roots
        for arg in config.args
    )


def pytest_addoption(parser):
    parser.addoption(
        "--test-profile",
        choices=("base", "mandatory"),
        default=None,
        help="base permits named optional omissions; unfiltered suites default mandatory",
    )


def pytest_configure(config):
    config.pluginmanager.register(_ProfileChecks(config), "plato-profile-checks")


class _ProfileChecks:
    """Enforce the two declared profiles and their exact omission allowances."""

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
        self.native = False
        self.violations = []
        self.passed = set()
        self.allowed_skips = set()

    def pytest_sessionstart(self):
        if not self.qualify or self.base:
            return
        try:
            for name in ("opacus", "kazoo", "gymnasium", "tiktoken"):
                importlib.import_module(name)
            with nanochat_source() as source:
                repository = Path(__file__).resolve().parents[1]
                expected_pin = subprocess.check_output(
                    ["git", "ls-tree", "HEAD", "external/nanochat"],
                    cwd=repository,
                    text=True,
                ).split()[2]
                actual_pin = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=source,
                    text=True,
                ).strip()
                if actual_pin != expected_pin:
                    raise ImportError(
                        "Nanochat checkout does not match its gitlink pin"
                    )
                model = importlib.import_module("nanochat.gpt")
                filename = getattr(model, "__file__", None)
                if filename is None or (
                    Path(filename).resolve()
                    != (source / "nanochat/gpt.py").resolve()
                ):
                    raise ImportError(
                        "Nanochat model must import from the pinned source"
                    )
                importlib.import_module("plato.models.nanochat")
                importlib.import_module("plato.trainers.nanochat")
            self.native = native_tokenizer_available()
        except (ImportError, subprocess.CalledProcessError) as exc:
            raise pytest.UsageError(f"mandatory prerequisite failed: {exc}") from exc

    def pytest_collection_modifyitems(self, items):
        for item in items:
            if not self.qualify:
                continue
            if self.base and item.nodeid in _NANOCHAT_TESTS:
                item.add_marker(pytest.mark.skip(reason=_BASE_NANOCHAT_REASON))
            elif (
                item.nodeid
                == _NANOCHAT + "test_nanochat_tokenizer_processor_round_trip"
                and not self.native
            ):
                item.add_marker(pytest.mark.skip(reason=_NATIVE_REASON))

    def _check_skip(self, report):
        if not self.qualify or not report.skipped:
            return
        reason = report.longrepr[2].removeprefix("Skipped: ")
        allowed = (
            (
                self.base
                and report.nodeid == _DP_MODULE
                and reason == _DP_REASON
                and importlib.util.find_spec("opacus") is None
            )
            or (
                self.base
                and report.nodeid in _NANOCHAT_TESTS
                and reason == _BASE_NANOCHAT_REASON
            )
            or (
                not self.base
                and not self.native
                and report.nodeid
                == _NANOCHAT + "test_nanochat_tokenizer_processor_round_trip"
                and reason == _NATIVE_REASON
            )
        )
        if allowed:
            self.allowed_skips.add(report.nodeid)
        else:
            self.violations.append(f"unexpected skip: {report.nodeid}: {reason}")

    def pytest_collectreport(self, report):
        self._check_skip(report)

    def pytest_runtest_logreport(self, report):
        if not self.qualify:
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
            if not (
                (partition == "runtime" and not runtime)
                or (partition == "not runtime" and runtime)
            ):
                self.violations.append(f"unexpected deselection: {item.nodeid}")

    def pytest_sessionfinish(self, session, exitstatus):
        if not self.qualify:
            return
        if any(
            self.config.getoption(option)
            for option in ("ignore", "ignore_glob", "deselect")
        ):
            self.violations.append("test profiles do not permit collection exclusions")
        if self.full:
            nodes = {item.nodeid for item in session.items}
            partition = self.config.getoption("markexpr")
            runtime = partition == "runtime"
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
        if self.violations:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter):
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
def initialized_nanochat(tmp_path, monkeypatch):
    """Keep the real source importable only for the duration of a Nanochat test."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "nanochat-cache"))
    with nanochat_source() as source:
        yield source


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
