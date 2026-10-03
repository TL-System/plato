"""Actual 8/24-sample native LeNet training over loopback sockets."""

import socket

import numpy as np

from plato.serialization.safetensor import deserialize_tree
from tests.mlx_native.helpers import assert_tree_equal
from tests.mlx_native.socket_harness import run_probe
from tests.mlx_native.test_phase3_runtime import paired_reference


def read_event(root, event):
    return deserialize_tree((root / event["weights_file"]).read_bytes())


def test_real_native_lenet_two_client_socket_round(tmp_path):
    result = run_probe(tmp_path / "socket", "socket_native", timeout=120)
    assert result["returncode"] == 0, result["stderr"]
    assert not result["timed_out"] and not result["forced"]
    assert not result["survivors"]
    events = result["events"]
    assert sum(e["event"] == "child_returned" for e in events) == 2
    # The inherited server close commits checkpoints and calls os._exit(0).
    # Its natural zero exit and both child returns are asserted above.
    assert sum(e["event"] == "server_close_started" for e in events) == 1
    baseline_event = next(e for e in events if e["event"] == "server_baseline")
    baseline = read_event(tmp_path / "socket", baseline_event)
    received = sorted(
        [e for e in events if e["event"] == "server_received"],
        key=lambda e: e["client_id"],
    )
    assert [(e["client_id"], e["samples"]) for e in received] == [(1, 8), (2, 24)]
    for e in [e for e in events if e["event"] == "client_baseline"]:
        assert_tree_equal(read_event(tmp_path / "socket", e), baseline)
    for e in [e for e in events if e["event"] == "client_trained"]:
        assert np.isfinite(e["loss"])
        assert e["device"] == "Device(gpu, 0)"
        trained = read_event(tmp_path / "socket", e)
        assert not np.array_equal(trained["fc3"]["weight"], baseline["fc3"]["weight"])
        arrival = next(r for r in received if r["client_id"] == e["client_id"])
        assert_tree_equal(read_event(tmp_path / "socket", arrival), trained)
    actual = read_event(
        tmp_path / "socket", next(e for e in events if e["event"] == "server_aggregate")
    )
    expected = paired_reference(*(read_event(tmp_path / "socket", e) for e in received))
    assert_tree_equal(actual, expected, rtol=1e-5, atol=1e-6)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", result["port"]))
