"""Startup behavior through fresh interpreter processes, with real asyncio."""

import sys

import numpy as np
import pytest

from tests.integration.startup_harness import run_probe

BOOTSTRAPS = [
    ("client_default", "client_body"),
    ("client_custom", "client_body"),
    ("ordinary", "periodic_body"),
    ("central", "periodic_body"),
    ("direct", "server_body"),
    ("edge_default", "client_body"),
    ("edge_custom", "client_body"),
]


def assert_natural(result):
    assert not result["timed_out"], result
    assert result["forced"] == [], result
    assert result["survivors"] == [], result
    assert "was never awaited" not in result["stderr"], result
    assert "Task was destroyed" not in result["stderr"], result


@pytest.mark.parametrize("case,body", BOOTSTRAPS)
def test_fresh_entrypoint_executes_scheduled_work(tmp_path, case, body):
    result = run_probe(tmp_path / case, case)
    assert_natural(result)
    assert result["returncode"] == 0, result
    assert any(e["event"] == body for e in result["events"]), result
    closed = sys.version_info >= (3, 14) or case == "direct"
    assert all(
        e["closed"] == closed for e in result["events"] if e["event"] == "loop_after"
    ), result
    if case in ("ordinary", "central"):
        assert sum(e["event"] == "periodic_body" for e in result["events"]) == 1
    if case.startswith("edge"):
        work = [
            e for e in result["events"] if e["event"] in ("server_work", "client_body")
        ]
        assert len({e["loop"] for e in work}) == 1, result
        assert not any(e["event"] == "periodic_body" for e in result["events"])
    if case == "edge_custom":
        assert any(
            e["event"] == "server_construct" and e["trainer"] == "trainer-factory"
            for e in result["events"]
        ), result


@pytest.mark.parametrize(
    "case,message",
    [
        ("client_custom", "client-sentinel"),
        ("ordinary", "periodic-sentinel"),
        ("central", "periodic-sentinel"),
        ("edge_custom", "client-sentinel"),
    ],
)
def test_active_failure_reaches_synchronous_caller(tmp_path, case, message):
    result = run_probe(tmp_path / case, case, fail=True)
    assert_natural(result)
    errors = [e for e in result["events"] if e["event"] == "caller_error"]
    assert result["returncode"] != 0, result
    assert [(e["type"], e["message"]) for e in errors] == [
        ("SentinelError", message)
    ], result


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("case", ["client_lifecycle", "ordinary"])
def test_preconstructed_borrowed_loop_resources_survive(tmp_path, case, fail):
    options = {"fail": fail} if case == "client_lifecycle" else {"start_fail": fail}
    result = run_probe(tmp_path / case, case, borrowed=True, **options)
    assert_natural(result)
    assert result["returncode"] == int(fail), result
    assert any(e["event"] == "borrowed_preserved" for e in result["events"]), result
    assert all(not e["closed"] for e in result["events"] if e["event"] == "loop_after")


@pytest.mark.parametrize("fail", [False, True])
def test_owned_client_finalizes_tasks_generators_and_executor(tmp_path, fail):
    result = run_probe(tmp_path / "client", "client_lifecycle", absent=True, fail=fail)
    assert_natural(result)
    assert result["returncode"] == int(fail), result
    events = {e["event"] for e in result["events"]}
    assert {"task_finalized", "generator_finalized", "executor_finished"} <= events
    assert all(e["closed"] for e in result["events"] if e["event"] == "loop_after")


def test_repeated_owned_client_entrypoints(tmp_path):
    result = run_probe(
        tmp_path / "client", "client_lifecycle", absent=True, repeat=True
    )
    assert_natural(result)
    assert result["returncode"] == 0, result
    assert sum(e["event"] == "client_body" for e in result["events"]) == 2
    assert sum(e["event"] == "generator_finalized" for e in result["events"]) == 2


@pytest.mark.parametrize("case", ["ordinary", "central", "edge_default", "edge_custom"])
def test_explicitly_absent_loop_on_server_and_edge(tmp_path, case):
    result = run_probe(tmp_path / case, case, absent=True)
    assert_natural(result)
    assert result["returncode"] == 0, result
    assert all(e["closed"] for e in result["events"] if e["event"] == "loop_after")


def test_closed_current_client_loop_is_replaced(tmp_path):
    result = run_probe(tmp_path / "closed", "client_custom", closed=True)
    assert_natural(result)
    assert result["returncode"] == 0, result
    assert any(e["event"] == "client_body" for e in result["events"]), result
    assert all(e["closed"] for e in result["events"] if e["event"] == "loop_after")


def test_running_loop_rejected_before_entrypoint_resources(tmp_path):
    result = run_probe(tmp_path / "running", "running")
    assert_natural(result)
    assert result["returncode"] == 0, result
    assert sum(e["event"] == "running_rejected" for e in result["events"]) == 3
    assert not any(e["event"] == "unexpected_resource" for e in result["events"])


def test_delegating_server_consumes_same_borrowed_loop(tmp_path):
    result = run_probe(tmp_path / "delegated", "ordinary", borrowed=True, real=True)
    assert_natural(result)
    assert result["returncode"] == 0, result
    assert any(e["event"] == "borrowed_consumed" for e in result["events"]), result
    loops = {e["loop"] for e in result["events"] if "loop" in e}
    assert len(loops) == 1, result


@pytest.mark.parametrize("start_fail", [False, True])
def test_owned_server_fallback_finalizes_its_resources(tmp_path, start_fail):
    result = run_probe(
        tmp_path / "fallback",
        "ordinary",
        absent=True,
        fallback_resources=True,
        start_fail=start_fail,
    )
    assert_natural(result)
    assert result["returncode"] == int(start_fail), result
    assert {"task_finalized", "generator_finalized", "executor_finished"} <= {
        e["event"] for e in result["events"]
    }, result
    assert all(e["closed"] for e in result["events"] if e["event"] == "loop_after")


def test_cancellation_already_finalizing_is_not_interrupted(tmp_path):
    result = run_probe(
        tmp_path / "cancelling", "ordinary", absent=True, self_cancel=True
    )
    assert_natural(result)
    assert result["returncode"] == 0, result
    assert any(e["event"] == "task_finalized" for e in result["events"]), result


def test_runner_cleanup_error_keeps_original_client_exception(tmp_path):
    result = run_probe(
        tmp_path / "cleanup",
        "client_lifecycle",
        absent=True,
        fail=True,
        cleanup_fail=True,
    )
    assert_natural(result)
    (error,) = [e for e in result["events"] if e["event"] == "caller_error"]
    assert error["message"] == "client-sentinel", result
    assert any("runner-cleanup-sentinel" in note for note in error["notes"]), result


def test_configure_failure_is_preserved_before_main_coroutine(tmp_path):
    result = run_probe(
        tmp_path / "configure", "client_lifecycle", absent=True, configure_fail=True
    )
    assert_natural(result)
    (error,) = [e for e in result["events"] if e["event"] == "caller_error"]
    assert error["message"] == "configure-sentinel", result
    assert not any(e["event"] == "client_body" for e in result["events"])


def test_missing_external_edge_client_preserves_value_error(tmp_path):
    result = run_probe(
        tmp_path / "edge", "edge_custom", absent=True, missing_client=True
    )
    assert_natural(result)
    (error,) = [e for e in result["events"] if e["event"] == "caller_error"]
    assert error["type"] == "ValueError", result
    assert error["message"] == "edge_client must be provided when edge_server is set."


@pytest.mark.parametrize(
    "control,code,kind",
    [("interrupt", -2, "KeyboardInterrupt"), ("exit", 7, "SystemExit")],
)
def test_client_control_flow_keeps_original_exception(tmp_path, control, code, kind):
    result = run_probe(
        tmp_path / control, "client_lifecycle", absent=True, control=control
    )
    assert_natural(result)
    assert result["returncode"] == code, result
    (error,) = [e for e in result["events"] if e["event"] == "caller_error"]
    assert error["type"] == kind, result


def test_active_edge_failure_unwinds_real_aiohttp(tmp_path):
    result = run_probe(
        tmp_path / "edge", "edge_custom", absent=True, real=True, fail=True
    )
    assert_natural(result)
    (error,) = [e for e in result["events"] if e["event"] == "caller_error"]
    assert error["message"] == "client-sentinel", result


def test_edge_cancellation_error_keeps_independent_setup_error(tmp_path):
    result = run_probe(
        tmp_path / "edge",
        "edge_custom",
        absent=True,
        real=True,
        startup_fail=True,
        secondary=True,
    )
    assert_natural(result)
    (error,) = [e for e in result["events"] if e["event"] == "caller_error"]
    assert error["message"] == "setup-sentinel", result
    assert any(e["event"] == "task_finalized" for e in result["events"]), result
    assert "client-finalizer" in result["stderr"], result


def test_independent_runtime_error_is_not_mistaken_for_monitor_stop(tmp_path):
    result = run_probe(
        tmp_path / "independent",
        "ordinary",
        absent=True,
        fail=True,
        same_message_error=True,
    )
    assert_natural(result)
    (error,) = [e for e in result["events"] if e["event"] == "caller_error"]
    assert error["type"] == "RuntimeError", result
    assert error["message"] == "Event loop stopped before Future completed.", result


@pytest.mark.parametrize(
    "options,message",
    [
        ({"real": True, "startup_fail": True, "secondary": True}, "setup-sentinel"),
        ({"real": True, "bind": True, "secondary": True}, None),
        ({"real": True, "fail": True}, "periodic-sentinel"),
        ({"start_fail": True, "fail": True}, "start-sentinel"),
        ({"real": True, "secondary": True}, "periodic-finalizer"),
    ],
)
def test_server_error_precedence_and_cooperative_teardown(tmp_path, options, message):
    result = run_probe(tmp_path / "server", "ordinary", absent=True, **options)
    assert_natural(result)
    assert result["returncode"] == 1, result
    (error,) = [e for e in result["events"] if e["event"] == "caller_error"]
    if message:
        assert error["message"] == message, result
    else:
        assert (
            error["type"] == "OSError" and "address already in use" in error["message"]
        )
    if options.get("secondary"):
        assert any(e["event"] == "task_finalized" for e in result["events"]), result
        assert "periodic-finalizer" in result["stderr"], result
    assert all(e["closed"] for e in result["events"] if e["event"] == "loop_after")


@pytest.mark.parametrize("stall", ["task", "generator", "executor", "aiohttp_task"])
def test_noncooperative_teardown_is_watchdog_containment(tmp_path, stall):
    if stall == "aiohttp_task":
        result = run_probe(
            tmp_path / stall,
            "ordinary",
            absent=True,
            real=True,
            startup_fail=True,
            secondary=True,
            stall="task",
        )
        stage = "task_finalizer_entered"
        original = "setup-sentinel"
    else:
        result = run_probe(
            tmp_path / stall, "client_lifecycle", absent=True, stall=stall
        )
        stage = (
            f"{stall}_finalizer_entered" if stall != "executor" else "executor_entered"
        )
        original = "client-sentinel"
    assert result["timed_out"] and result["forced"], result
    assert result["survivors"] == [], result
    assert result["elapsed"] < 40, result
    assert any(e["event"] == stage for e in result["events"]), result
    assert any(
        e["event"] == "original_failure" and e["message"] == original
        for e in result["events"]
    ), result
    assert not any(e["event"] == "caller_error" for e in result["events"]), result


@pytest.mark.integration
def test_real_two_client_cpu_socket_round(tmp_path):
    result = run_probe(tmp_path / "round", "socket_round", timeout=90)
    assert_natural(result)
    assert result["returncode"] == 0, result
    events = result["events"]
    assert any(e["event"] == "server_close_started" for e in events), result
    (initial,) = [e["weights"] for e in events if e["event"] == "server_baseline"]
    (final,) = [e["weights"] for e in events if e["event"] == "server_aggregate"]
    trained = {e["client_id"]: e for e in events if e["event"] == "client_trained"}
    received = {e["client_id"]: e for e in events if e["event"] == "server_received"}
    client_initial = {
        e["client_id"]: e["weights"] for e in events if e["event"] == "client_baseline"
    }
    assert set(trained) == set(received) == set(client_initial) == {1, 2}, result
    assert len({e["pid"] for e in trained.values()}) == 2, result
    assert sorted(e["samples"] for e in received.values()) == [2, 6]
    assert {e["pid"] for e in events if e["event"] == "child_returned"} == {
        e["pid"] for e in trained.values()
    }, result
    for client_id, observation in trained.items():
        assert observation["device"] == "cpu", result
        assert observation["samples"] == received[client_id]["samples"]
        assert any(
            not np.allclose(observation["weights"][key], initial[key])
            for key in initial
        ), result
        for key in initial:
            np.testing.assert_allclose(client_initial[client_id][key], initial[key])
            np.testing.assert_allclose(
                received[client_id]["weights"][key], observation["weights"][key]
            )
    for key, baseline in initial.items():
        baseline = np.asarray(baseline)
        reference = (
            baseline
            + sum(
                e["samples"] * (np.asarray(e["weights"][key]) - baseline)
                for e in received.values()
            )
            / 8
        )
        np.testing.assert_allclose(final[key], reference, rtol=1e-5, atol=1e-6)
    import socket

    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", result["port"]))
        sock.listen()


def test_post_launch_failure_keeps_primary_error_and_contains_children(tmp_path):
    result = run_probe(tmp_path / "failure", "socket_failure", timeout=30)
    assert not result["timed_out"] and not result["survivors"], result
    (error,) = [e for e in result["events"] if e["event"] == "caller_error"]
    assert error["message"] == "post-launch-sentinel", result
    assert sum(e["event"] == "child_started" for e in result["events"]) == 2


def test_stalled_real_client_is_contained_without_round_success(tmp_path):
    result = run_probe(tmp_path / "stalled", "socket_stall", timeout=90)
    assert result["timed_out"] and result["forced"], result
    assert result["survivors"] == [], result
    assert result["elapsed"] < 100, result
    assert any(e["event"] == "client_stall_entered" for e in result["events"]), result
    assert not any(e["event"] == "server_aggregate" for e in result["events"]), result
    (processed,) = [
        e for e in result["events"] if e["event"] == "server_payload_processed"
    ]
    assert processed["client_id"] == 1, result
    (other,) = [
        e
        for e in result["events"]
        if e["event"] == "client_trained" and e["client_id"] == 1
    ]
    assert processed["samples"] == other["samples"], result
    for name, weights in processed["weights"].items():
        np.testing.assert_allclose(weights, other["weights"][name])
