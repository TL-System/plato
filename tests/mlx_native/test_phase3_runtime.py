"""Real trained LeNets through guarded aggregation, codecs and checkpoints."""

import asyncio
import copy

import mlx.core as mx
import numpy as np
import pytest

from plato.algorithms.mlx_fedavg import Algorithm
from plato.callbacks.server import ServerCallback
from plato.models.mlx.lenet5 import LeNet5
from plato.processors.safetensor_decode import Processor as Decode
from plato.processors.safetensor_encode import Processor as Encode
from plato.servers.strategies.aggregation import FedAvgAggregationStrategy
from plato.trainers.mlx import ComposableMLXTrainer, _tree_leaves
from tests.mlx_native.helpers import (
    assert_tree_equal,
    dataset,
    native_config,
    update,
)


class DeltaOnly(FedAvgAggregationStrategy):
    async def aggregate_deltas(self, updates, deltas_received, context):
        context.state["arithmetic_called"] = True
        return await super().aggregate_deltas(updates, deltas_received, context)


class ObservedWeights(FedAvgAggregationStrategy):
    async def aggregate_weights(self, updates, baseline, weights, context):
        context.state["arithmetic_called"] = True
        return await super().aggregate_weights(updates, baseline, weights, context)


def server_for(trainer, kind):
    from plato.servers import fedavg

    class Legacy(fedavg.Server):
        async def aggregate_weights(self, updates, baseline, weights):
            self.context.state["arithmetic_called"] = True
            return await FedAvgAggregationStrategy().aggregate_weights(
                updates, baseline, weights, self.context
            )

    server = (Legacy if kind == "legacy" else fedavg.Server)(
        aggregation_strategy=DeltaOnly()
        if kind in ("delta", "legacy")
        else ObservedWeights()
    )
    server.trainer = trainer
    server.algorithm = Algorithm(trainer)
    server.context.trainer = trainer
    server.context.algorithm = server.algorithm
    server.context.server = server
    server.clients_processed = lambda: None
    return server


@pytest.mark.parametrize("kind", ["direct", "delta", "legacy"])
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["valid-first", "malformed-first"]
)
@pytest.mark.parametrize("boundary", ["raw", "hook", "callback"])
@pytest.mark.parametrize("bad_count", [24, 0], ids=["weighted", "zero-weight"])
@pytest.mark.parametrize(
    "damage", ["broadcast", "missing", "extra", "none", "container"]
)
def test_server_rejects_every_eligible_tree_before_arithmetic(
    tmp_path, kind, reverse, boundary, bad_count, damage
):
    with native_config(tmp_path):
        trainer = ComposableMLXTrainer(model=LeNet5)
        server = server_for(trainer, kind)
        baseline = server.algorithm.extract_weights()
        good = copy.deepcopy(baseline)
        bad = copy.deepcopy(baseline)
        if damage == "broadcast":
            bad["fc3"]["bias"] = np.zeros(1, dtype=np.float32)
        elif damage == "missing":
            del bad["fc3"]["bias"]
        elif damage == "extra":
            bad["fc3"]["extra"] = np.zeros(1, dtype=np.float32)
        elif damage == "none":
            bad["fc3"]["bias"] = None
        else:
            bad["fc3"] = list(bad["fc3"].values())
        server.updates = [
            update(1, 8, good),
            update(2, bad_count, bad if boundary == "raw" else good),
        ]
        if reverse:
            server.updates.reverse()
        original = copy.deepcopy([u.payload for u in server.updates])

        def corrupt(weights):
            server.context.state["hook_called"] = True
            weights[0 if reverse else 1] = copy.deepcopy(bad)
            return weights

        if boundary == "hook":
            server.weights_received = corrupt
        elif boundary == "callback":

            class Corrupt(ServerCallback):
                def on_weights_received(self, server, weights_received):
                    corrupt(weights_received)

            server.callback_handler.add_callback(Corrupt)
        with pytest.raises(ValueError, match="client 2.*fc3"):
            asyncio.run(server._process_reports())
        assert not server.context.state.get("arithmetic_called")
        if boundary == "raw":
            assert not server.context.state.get("hook_called")
        assert_tree_equal(server.algorithm.extract_weights(), baseline)
        assert_tree_equal([u.payload for u in server.updates], original)


@pytest.mark.parametrize("kind", ["direct", "delta", "legacy"])
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["valid-first", "malformed-first"]
)
@pytest.mark.parametrize("damage", ["short", "long", "tuple"])
def test_server_exact_native_sequence_tree(tmp_path, kind, reverse, damage):
    import mlx.nn as nn

    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(2, 2), nn.Linear(2, 2)]

    with native_config(tmp_path):
        trainer = ComposableMLXTrainer(model=Nested)
        server = server_for(trainer, kind)
        baseline = server.algorithm.extract_weights()
        malformed = copy.deepcopy(baseline)
        if damage == "short":
            malformed["layers"].pop()
        elif damage == "long":
            malformed["layers"].append(copy.deepcopy(malformed["layers"][0]))
        else:
            malformed["layers"] = tuple(malformed["layers"])
        server.updates = [update(1, 8, baseline), update(2, 24, malformed)]
        if reverse:
            server.updates.reverse()
        with pytest.raises(ValueError, match="client 2.*layers"):
            asyncio.run(server._process_reports())
        assert not server.context.state.get("arithmetic_called")
        assert_tree_equal(server.algorithm.extract_weights(), baseline)


@pytest.mark.parametrize("kind", ["direct", "delta", "legacy"])
def test_server_rejects_observable_hook_reordering(tmp_path, kind):
    with native_config(tmp_path):
        trainer = ComposableMLXTrainer(model=LeNet5)
        server = server_for(trainer, kind)
        baseline = server.algorithm.extract_weights()
        server.updates = [
            update(1, 8, copy.deepcopy(baseline)),
            update(2, 24, copy.deepcopy(baseline)),
        ]
        server.weights_received = lambda weights: list(reversed(weights))
        with pytest.raises(ValueError, match="reordered"):
            asyncio.run(server._process_reports())
        assert not server.context.state.get("arithmetic_called")
        assert_tree_equal(server.algorithm.extract_weights(), baseline)


@pytest.mark.parametrize("kind", ["direct", "delta", "legacy"])
@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_actual_unequal_client_training_matches_float64_weighted_reference(
    tmp_path, kind, optimizer
):
    with native_config(tmp_path, optimizer=optimizer, model_seed=17, training_seed=29):
        server_trainer = ComposableMLXTrainer(model=LeNet5)
        baseline = Algorithm(server_trainer).extract_weights()
        retained = copy.deepcopy(baseline)
        trained = []
        for client_id, count in ((1, 8), (2, 24)):
            trainer = ComposableMLXTrainer(model=LeNet5)
            trainer.set_client_id(client_id)
            algorithm = Algorithm(trainer)
            algorithm.load_weights(baseline)
            trainer.train_model(
                trainer_config(trainer), dataset(count, client_id), None
            )
            assert np.isfinite(trainer.context.state["last_loss"])
            weights = algorithm.extract_weights()
            assert any(
                not np.array_equal(a, b)
                for a, b in zip(
                    _tree_leaves(weights), _tree_leaves(baseline), strict=True
                )
            )
            trained.append(Decode().process(Encode().process(weights)))
        assert_tree_equal(baseline, retained)
        reference = paired_reference(trained[0], trained[1])
        server = server_for(server_trainer, kind)
        server.updates = [update(1, 8, trained[0]), update(2, 24, trained[1])]
        asyncio.run(server._process_reports())
        assert_tree_equal(
            server.algorithm.extract_weights(), reference, rtol=1e-5, atol=1e-6
        )


def trainer_config(trainer):
    from plato.config import Config

    return Config().trainer._asdict()


def paired_reference(first, second):
    if isinstance(first, dict):
        return {k: paired_reference(first[k], second[k]) for k in first}
    if isinstance(first, (list, tuple)):
        return type(first)(
            paired_reference(a, b) for a, b in zip(first, second, strict=True)
        )
    if first is None:
        return None
    return (first.astype(np.float64) * 0.25 + second.astype(np.float64) * 0.75).astype(
        first.dtype
    )


def test_native_checkpoint_codec_prediction_equality_and_next_update(tmp_path):
    with native_config(tmp_path, model_seed=17, training_seed=29):
        trainer = ComposableMLXTrainer(model=LeNet5)
        algorithm = Algorithm(trainer)
        samples = dataset()
        trainer.train_model(trainer_config(trainer), samples, None)
        snapshot = algorithm.extract_weights()
        batch = mx.array(np.stack([x for x, _ in samples]))
        prediction = np.array(trainer.model(batch), copy=True)
        trainer.save_model("round.safetensors", str(tmp_path))
        restored = ComposableMLXTrainer(model=LeNet5)
        restored.load_model("round.safetensors", str(tmp_path))
        assert_tree_equal(Algorithm(restored).extract_weights(), snapshot)
        np.testing.assert_array_equal(np.asarray(restored.model(batch)), prediction)
        decoded = Decode().process(Encode().process(snapshot))
        assert_tree_equal(decoded, snapshot)
        malformed = copy.deepcopy(snapshot)
        malformed["fc3"]["bias"] = np.zeros(1, dtype=np.float32)
        from plato.serialization.safetensor import serialize_tree

        (tmp_path / "bad.safetensors").write_bytes(serialize_tree(malformed))
        with pytest.raises(ValueError, match="fc3.bias"):
            restored.load_model("bad.safetensors", str(tmp_path))
        assert_tree_equal(Algorithm(restored).extract_weights(), snapshot)
        restored.train_model(trainer_config(restored), samples, None)
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(
                _tree_leaves(Algorithm(restored).extract_weights()),
                _tree_leaves(snapshot),
                strict=True,
            )
        )


def test_float16_transport_and_explicit_bfloat16_rejection():
    from plato.trainers.mlx import _to_host_array

    host = _to_host_array(mx.array([1, 2], dtype=mx.float16))
    assert host.dtype == np.float16
    assert_tree_equal(Decode().process(Encode().process({"x": host})), {"x": host})
    with pytest.raises(ValueError, match="bfloat16"):
        _to_host_array(mx.array([1, 2], dtype=mx.bfloat16))


def test_cpu_and_metal_equal_work_training_agree_with_tolerance(tmp_path):
    from plato.config import Config

    with native_config(tmp_path, model_seed=17, training_seed=29):
        trainers = []
        baseline = None
        for cpu in (True, False):
            Config.args.cpu, Config.args.mps = cpu, not cpu
            trainer = ComposableMLXTrainer(model=LeNet5)
            algorithm = Algorithm(trainer)
            if baseline is None:
                baseline = algorithm.extract_weights()
            algorithm.load_weights(baseline)
            trainer.set_client_id(1)
            trainer.train_model(trainer_config(trainer), dataset(), None)
            trainers.append(trainer)
        assert_tree_equal(
            Algorithm(trainers[0]).extract_weights(),
            Algorithm(trainers[1]).extract_weights(),
            rtol=1e-5,
            atol=1e-6,
        )
        assert trainers[0].context.state["last_loss"] == pytest.approx(
            trainers[1].context.state["last_loss"], rel=1e-5, abs=1e-6
        )
