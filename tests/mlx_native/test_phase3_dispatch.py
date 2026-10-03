"""F17 preflight must follow the last mutable report-selection callback."""

import asyncio
import copy

import mlx.core as mx
import numpy as np
import pytest

from plato.models.mlx.lenet5 import LeNet5
from plato.trainers.mlx import ComposableMLXTrainer
from tests.mlx_native.helpers import assert_tree_equal, native_config, update
from tests.mlx_native.test_phase3_runtime import paired_reference, server_for


def prepared_server(kind, reverse):
    trainer = ComposableMLXTrainer(model=LeNet5)
    server = server_for(trainer, kind)
    baseline = server.algorithm.extract_weights()
    clients = [copy.deepcopy(baseline), copy.deepcopy(baseline)]
    clients[0]["fc3"]["bias"] = np.full(10, 1, dtype=np.float32)
    clients[1]["fc3"]["bias"] = np.full(10, 4, dtype=np.float32)
    server.updates = [
        update(1, 8, copy.deepcopy(clients[0])),
        update(2, 24, copy.deepcopy(clients[1])),
    ]
    if reverse:
        server.updates.reverse()
    compute = server.algorithm.compute_weight_deltas

    def observe_subtraction(*args, **kwargs):
        server.context.state["delta_dispatch_called"] = True
        return compute(*args, **kwargs)

    server.algorithm.compute_weight_deltas = observe_subtraction
    return server, baseline, clients


def rejected_without_arithmetic(server, baseline, clients, match):
    retained_clients = copy.deepcopy(clients)
    retained_baseline = copy.deepcopy(baseline)
    device = mx.default_device()
    streams = (mx.default_stream(mx.cpu), mx.default_stream(mx.gpu))
    hook_state = {}
    receive = server.weights_received
    select = server.client_selection_strategy.on_reports_received

    def remember_received(weights):
        effective = receive(weights)
        hook_state["effective"] = effective
        return effective

    def remember_selected(updates, context):
        select(updates, context)
        # Deliberate hook side effects remain; preflight must add no mutations.
        hook_state["trees_after_hook"] = copy.deepcopy(hook_state["effective"])
        hook_state["payloads_after_hook"] = copy.deepcopy(
            [received.payload for received in server.updates]
        )
        hook_state["clients_after_hook"] = [
            received.client_id for received in server.updates
        ]
        hook_state["counts_after_hook"] = [
            received.report.num_samples for received in server.updates
        ]

    server.weights_received = remember_received
    server.client_selection_strategy.on_reports_received = remember_selected
    with pytest.raises(ValueError, match=match):
        asyncio.run(server._process_reports())
    assert not server.context.state.get("arithmetic_called")
    assert not server.context.state.get("delta_dispatch_called")
    assert_tree_equal(server.algorithm.extract_weights(), retained_baseline)
    assert_tree_equal(baseline, retained_baseline)
    assert_tree_equal(clients, retained_clients)
    assert_tree_equal(hook_state["effective"], hook_state["trees_after_hook"])
    assert_tree_equal(
        [received.payload for received in server.updates],
        hook_state["payloads_after_hook"],
    )
    assert [received.client_id for received in server.updates] == (
        hook_state["clients_after_hook"]
    )
    np.testing.assert_array_equal(
        [received.report.num_samples for received in server.updates],
        hook_state["counts_after_hook"],
    )
    assert mx.default_device() == device
    assert (mx.default_stream(mx.cpu), mx.default_stream(mx.gpu)) == streams


@pytest.mark.parametrize("kind", ["direct", "delta", "legacy"])
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["client1-first", "client2-first"]
)
@pytest.mark.parametrize("bad_count", [24, 0], ids=["weighted", "zero-weight"])
@pytest.mark.parametrize("damage", ["broadcast", "missing", "none"])
def test_late_selector_rejects_malformed_tree_before_arithmetic(
    tmp_path, kind, reverse, bad_count, damage
):
    with native_config(tmp_path, model_seed=17):
        server, baseline, clients = prepared_server(kind, reverse)
        malformed = next(u for u in server.updates if u.client_id == 2)
        malformed.report.num_samples = bad_count

        def corrupt(updates, context):
            context.state["selector_called"] = True
            if damage == "broadcast":
                malformed.payload["fc3"]["bias"] = np.full(1, 4, dtype=np.float32)
            elif damage == "missing":
                del malformed.payload["fc3"]["bias"]
            else:
                malformed.payload["fc3"]["bias"] = None

        server.client_selection_strategy.on_reports_received = corrupt
        rejected_without_arithmetic(server, baseline, clients, "client 2.*fc3")
        assert server.context.state["selector_called"]


@pytest.mark.parametrize("kind", ["direct", "delta", "legacy"])
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["client1-first", "client2-first"]
)
@pytest.mark.parametrize("damage", ["updates", "reports", "client-ids"])
def test_late_selector_rejects_observable_report_reassociation(
    tmp_path, kind, reverse, damage
):
    with native_config(tmp_path, model_seed=17):
        server, baseline, clients = prepared_server(kind, reverse)

        def reorder(updates, context):
            context.state["selector_called"] = True
            if damage == "updates":
                updates.reverse()
            elif damage == "reports":
                updates[0].report, updates[1].report = (
                    updates[1].report,
                    updates[0].report,
                )
            else:
                updates[0].client_id, updates[1].client_id = (
                    updates[1].client_id,
                    updates[0].client_id,
                )

        server.client_selection_strategy.on_reports_received = reorder
        rejected_without_arithmetic(server, baseline, clients, "reordered|association")
        assert server.context.state["selector_called"]


@pytest.mark.parametrize("kind", ["direct", "delta", "legacy"])
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["client1-first", "client2-first"]
)
@pytest.mark.parametrize(
    "damage", ["short", "long", "negative", "nan", "infinite", "total-overflow"]
)
def test_late_selector_revalidates_report_counts_before_arithmetic(
    tmp_path, kind, reverse, damage
):
    with native_config(tmp_path, model_seed=17):
        server, baseline, clients = prepared_server(kind, reverse)

        def corrupt(updates, context):
            context.state["selector_called"] = True
            if damage == "short":
                updates.pop()
            elif damage == "long":
                updates.append(copy.deepcopy(updates[-1]))
            elif damage == "total-overflow":
                for received in updates:
                    received.report.num_samples = 1e308
            else:
                value = {"negative": -1, "nan": float("nan"), "infinite": float("inf")}
                updates[-1].report.num_samples = value[damage]

        server.client_selection_strategy.on_reports_received = corrupt
        rejected_without_arithmetic(
            server, baseline, clients, "payload|sample|positive|weight"
        )
        assert server.context.state["selector_called"]


@pytest.mark.parametrize("kind", ["direct", "delta", "legacy"])
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["client1-first", "client2-first"]
)
def test_late_selector_preserves_valid_positional_copy_replacements(
    tmp_path, kind, reverse
):
    with native_config(tmp_path, model_seed=17):
        server, baseline, clients = prepared_server(kind, reverse)
        server.weights_received = copy.deepcopy

        def replace(updates, context):
            context.state["selector_called"] = True
            updates[:] = copy.deepcopy(updates)

        server.client_selection_strategy.on_reports_received = replace
        asyncio.run(server._process_reports())
        assert server.context.state["selector_called"]
        assert server.context.state["arithmetic_called"]
        assert_tree_equal(
            server.algorithm.extract_weights(),
            paired_reference(*clients),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            server.algorithm.extract_weights()["fc3"]["bias"],
            np.full(10, 3.25, dtype=np.float32),
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize("kind", ["direct", "delta"])
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["client1-first", "client2-first"]
)
def test_late_selector_preserves_valid_zero_total_weights(tmp_path, kind, reverse):
    with native_config(tmp_path, model_seed=17):
        server, baseline, clients = prepared_server(kind, reverse)

        def zero(updates, context):
            for received in updates:
                received.report.num_samples = 0

        server.client_selection_strategy.on_reports_received = zero
        asyncio.run(server._process_reports())
        assert_tree_equal(server.algorithm.extract_weights(), baseline)
