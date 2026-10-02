"""Pytest fixtures shared across test modules."""

import importlib
import importlib.util
import random
import sys
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


def pytest_addoption(parser):
    parser.addoption(
        "--test-profile",
        choices=("base", "mandatory"),
        default="mandatory",
        help="base permits only named optional omissions; mandatory requires extras",
    )


def pytest_configure(config):
    config.pluginmanager.register(_ProfileChecks(config), "plato-profile-checks")


class _ProfileChecks:
    """Enforce the two declared profiles and their exact omission allowances."""

    def __init__(self, config):
        self.config = config
        self.base = config.getoption("test_profile") == "base"
        self.native = False
        self.violations = []
        self.passed = set()
        self.allowed_skips = set()

    def pytest_sessionstart(self):
        if self.base:
            return
        try:
            for name in ("opacus", "kazoo", "gymnasium", "tiktoken"):
                importlib.import_module(name)
            with nanochat_source():
                importlib.import_module("nanochat.gpt")
                importlib.import_module("plato.models.nanochat")
                importlib.import_module("plato.trainers.nanochat")
            self.native = native_tokenizer_available()
        except ImportError as exc:
            raise pytest.UsageError(f"mandatory prerequisite failed: {exc}") from exc

    def pytest_collection_modifyitems(self, items):
        for item in items:
            if item.nodeid in _RUNTIME:
                item.add_marker(pytest.mark.runtime)
            if self.base and item.nodeid in _NANOCHAT_TESTS:
                item.add_marker(pytest.mark.skip(reason=_BASE_NANOCHAT_REASON))
            elif (
                item.nodeid
                == _NANOCHAT + "test_nanochat_tokenizer_processor_round_trip"
                and not self.native
            ):
                item.add_marker(pytest.mark.skip(reason=_NATIVE_REASON))

    def _check_skip(self, report):
        if not report.skipped:
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
        if hasattr(report, "wasxfail"):
            self.violations.append(f"unexpected xfail/xpass: {report.nodeid}")
        else:
            self._check_skip(report)
        if report.when == "call" and report.passed:
            self.passed.add(report.nodeid)

    def pytest_deselected(self, items):
        partition = self.config.getoption("markexpr")
        for item in items:
            runtime = item.get_closest_marker("runtime") is not None
            if not (
                (partition == "runtime" and not runtime)
                or (partition == "not runtime" and runtime)
            ):
                self.violations.append(f"unexpected deselection: {item.nodeid}")

    def pytest_sessionfinish(self, session, exitstatus):
        if any(
            self.config.getoption(option)
            for option in ("ignore", "ignore_glob", "deselect")
        ):
            self.violations.append("test profiles do not permit collection exclusions")
        full = any(
            Path(str(arg).split("::", 1)[0]).resolve()
            in (
                self.config.rootpath.resolve(),
                (self.config.rootpath / "tests").resolve(),
            )
            for arg in self.config.args
        )
        if full:
            nodes = {item.nodeid for item in session.items}
            runtime = self.config.getoption("markexpr") == "runtime"
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
