"""Main-guarded real-model probes; collection imports only the standard library."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, cast

REPOSITORY = Path(__file__).resolve().parents[2]


def emit(event: str, **fields: Any) -> None:
    """Persist observations before the server's process shutdown."""
    root = Path(os.environ["PLATO_RETENTION_ROOT"])
    data = {"event": event, "pid": os.getpid(), **fields}
    with (root / f"events-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(data) + "\n")


def family_module() -> Any:
    family = os.environ["PLATO_RETENTION_FAMILY"]
    directory = (
        REPOSITORY / "examples" / "gradient_leakage_attacks"
        if family == "dlg"
        else REPOSITORY / "examples" / "model_search" / family
    )
    sys.path.insert(0, str(directory))
    return importlib.import_module("dlg_model" if family == "dlg" else family)


def clone(weights: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in weights.items()}


def fixture(count: int, client_id: int = 0) -> Any:
    import torch
    from torch.utils.data import TensorDataset

    generator = torch.Generator().manual_seed(1234 + client_id)
    inputs = torch.randn(count, 3, 32, 32, generator=generator) * 0.2
    inputs += client_id * 0.1
    targets = (torch.arange(count) + client_id) % 10
    return TensorDataset(inputs, targets)


class TinyDatasource:
    """Deterministic CIFAR-shaped local data with realized 2/6 sample counts."""

    def __init__(self) -> None:
        from plato.config import Config

        client_id = Config().args.id or 0
        self.trainset = fixture(2 if client_id == 1 else 6, client_id)
        self.testset = self.trainset
        emit("local_data", client_id=client_id, samples=len(self.trainset))

    def get_train_set(self) -> Any:
        return self.trainset

    def get_test_set(self) -> Any:
        return self.testset

    def num_train_examples(self) -> int:
        return len(self.trainset)

    def num_test_examples(self) -> int:
        return len(self.testset)


class ClientObserver:
    """Observe actual payload replacement and learned weights without changing them."""

    def on_inbound_received(self, client: Any, inbound_processor: Any) -> None:
        pass

    def on_inbound_processed(self, client: Any, data: Any) -> None:
        model = client.trainer.model
        assert model is client.algorithm.model
        assert {
            key: list(value.shape) for key, value in model.state_dict().items()
        } == {key: list(value.shape) for key, value in data.items()}
        client._retention_baseline = clone(data)
        emit(
            "client_replaced_before_load",
            client_id=client.client_id,
            first_conv_channels=model.conv1.out_channels,
        )

    def on_outbound_ready(
        self, client: Any, report: Any, outbound_processor: Any
    ) -> None:
        import torch

        weights = client.algorithm.extract_weights()
        changed = sum(
            not torch.equal(value, client._retention_baseline[key])
            for key, value in weights.items()
            if "weight" in key or "bias" in key
        )
        assert changed > 0
        assert all(torch.isfinite(value).all() for value in weights.values())
        emit(
            "client_trained",
            client_id=client.client_id,
            samples=report.num_samples,
            changed_parameters=changed,
            shapes={key: list(v.shape) for key, v in weights.items()},
        )


class LossObserver:
    """Record real local training losses through the existing callback API."""

    def on_train_run_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_train_run_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_train_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_train_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_train_step_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_train_step_end(
        self, trainer: Any, config: Any, batch: Any, loss: Any, **kwargs: Any
    ) -> None:
        scalar = float(loss)
        assert math.isfinite(scalar)
        emit("training_loss", client_id=trainer.client_id, loss=scalar)

    def on_test_outputs(self, trainer: Any, outputs: Any, **kwargs: Any) -> Any:
        return outputs


def aggregate_reference(baseline: Any, payloads: Any) -> dict[str, Any]:
    """Independently index every shared channel with the family's uniform average."""
    import torch

    result = clone(baseline)
    for key, value in baseline.items():
        if "weight" not in key and "bias" not in key:
            continue
        total = torch.zeros_like(value)
        count = torch.zeros_like(value)
        for payload in payloads:
            local = payload[key]
            region = tuple(slice(0, length) for length in local.shape)
            total[region] += local
            count[region] += 1
        result[key] = torch.where(count > 0, total / count.clamp_min(1), value)
    return result


def channel_reference(weights: Any, *, rolling: bool) -> dict[str, Any]:
    """Index the known ResNet18 modules independently of the algorithm's key loop."""
    import torch

    result = clone(weights)

    def permutation(key: str) -> Any:
        value = weights[key]
        if rolling:
            return (torch.arange(value.shape[0]) - 1) % value.shape[0]
        norms = torch.linalg.vector_norm(value, dim=(1, 2, 3))
        return torch.argsort(norms, descending=True)

    def normalize(prefix: str, order: Any) -> None:
        for suffix in ("weight", "bias"):
            result[prefix + suffix] = weights[prefix + suffix].index_select(0, order)

    order = permutation("conv1.weight")
    result["conv1.weight"] = weights["conv1.weight"].index_select(0, order)
    for layer in range(1, 5):
        for block in range(2):
            prefix = f"layer{layer}.{block}."
            incoming = order
            normalize(prefix + "batchnorm1.", incoming)
            first = permutation(prefix + "conv1.weight")
            result[prefix + "conv1.weight"] = (
                weights[prefix + "conv1.weight"]
                .index_select(1, incoming)
                .index_select(0, first)
            )
            normalize(prefix + "batchnorm2.", first)
            order = permutation(prefix + "conv2.weight")
            result[prefix + "conv2.weight"] = (
                weights[prefix + "conv2.weight"]
                .index_select(1, first)
                .index_select(0, order)
            )
            shortcut = prefix + "shortcut.weight"
            if shortcut in weights:
                result[shortcut] = weights[shortcut].index_select(1, incoming)
                result[shortcut] = result[shortcut].index_select(0, order)
    normalize("bn4.", order)
    result["linear.weight"] = weights["linear.weight"].index_select(1, order)
    return result


def compare(actual: Any, expected: Any, *, parameters_only: bool = False) -> float:
    import torch

    errors = []
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if parameters_only and "weight" not in key and "bias" not in key:
            continue
        torch.testing.assert_close(actual[key], value, rtol=1e-5, atol=1e-6, msg=key)
        errors.append(float((actual[key].float() - value.float()).abs().max()))
    return max(errors, default=0.0)


def train_batch(model: Any) -> float:
    import torch

    model.train()
    baseline = clone(model.state_dict())
    inputs, labels = fixture(2).tensors
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    outputs = model(inputs)
    assert torch.isfinite(outputs).all()
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    optimizer.step()
    assert any(
        not torch.equal(value, baseline[key])
        for key, value in model.state_dict().items()
        if "weight" in key
    )
    return float(loss.detach())


class ServerObserver:
    """Assert numerical aggregation, alignment, running statistics and shutdown."""

    def on_training_will_start(self, server: Any, **kwargs: Any) -> None:
        server._retention_baseline = clone(server.algorithm.model.state_dict())
        emit(
            "server_trainer_initialized",
            channels=server.trainer.model.conv1.out_channels,
            model_rate_parameters=server.trainer.model.linear.in_features,
        )

    def on_clients_selected(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_clients_processed(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_weights_received(self, server: Any, weights_received: Any) -> None:
        counts = sorted(update.report.num_samples for update in server.updates)
        assert counts == [2, 6]
        server._retention_expected = aggregate_reference(
            server._retention_baseline, weights_received
        )
        emit("server_received", counts=counts, payloads=len(weights_received))

    def on_weights_aggregated(self, server: Any, updates: Any) -> None:
        import torch
        from torch.utils.data import DataLoader

        family = os.environ["PLATO_RETENTION_FAMILY"]
        expected = server._retention_presort
        if family != "heterofl":
            expected = channel_reference(expected, rolling=family == "fedrolex")
        error = compare(
            server.algorithm.model.state_dict(), expected, parameters_only=True
        )
        emit(
            "channel_alignment_oracle",
            max_error=error,
            parameter_tensors=sum("weight" in key or "bias" in key for key in expected),
            weighting="uniform average over clients containing each channel",
        )
        if family == "fedrolex":
            before = clone(server.algorithm.model.state_dict())
            expected_second = channel_reference(before, rolling=True)
            server.algorithm.sort_channels()
            second_error = compare(server.algorithm.model.state_dict(), expected_second)
            assert not torch.equal(
                before["conv1.weight"], expected_second["conv1.weight"]
            )
            emit(
                "second_rolling_alignment",
                max_error=second_error,
                first_conv_order=[63, *range(63)],
            )
        model = server.algorithm.model
        if family == "heterofl":
            model = server.algorithm.stat(
                server.model, DataLoader(fixture(6), batch_size=2)
            )
            statistics = [
                module
                for module in model.modules()
                if isinstance(module, torch.nn.BatchNorm2d)
            ]
            assert statistics and all(
                int(m.num_batches_tracked) >= 3 for m in statistics
            )
            assert all(torch.isfinite(m.running_mean).all() for m in statistics)
            assert any(torch.count_nonzero(m.running_mean) for m in statistics)
            emit(
                "sbn_statistics",
                batchnorm_modules=len(statistics),
                min_batches=min(int(m.num_batches_tracked) for m in statistics),
            )
        model.eval()
        with torch.no_grad():
            assert torch.isfinite(model(fixture(2).tensors[0])).all()
        loss = train_batch(copy.deepcopy(model))
        emit("post_alignment_backward", loss=loss)
        server._retention_verified = True

    def on_server_will_close(self, server: Any, **kwargs: Any) -> None:
        assert server._retention_verified and server.current_round == 1
        emit("success", round=server.current_round, real_socket=True)


def observe_client_run(*args: Any) -> None:
    from plato.client import run
    from plato.config import Config

    run(*args)
    emit("child_returned", client_id=Config().args.id)


def socket_round(module: Any) -> None:
    import multiprocessing as mp

    import psutil

    from plato.servers import base

    family = os.environ["PLATO_RETENTION_FAMILY"]
    model = module.resnet.resnet18 if family == "heterofl" else module.resnet18
    if family == "heterofl":
        callback_module = importlib.import_module("heterofl_trainer")
        callback_trainer = module.ServerTrainer(model=model)
        callback_trainer.model = model(model_rate=0.5, track=True)
        assert callback_trainer.model.conv1.out_channels == 32
        callback_module.ModelReinitializationCallback(model).on_trainer_initialized(
            callback_trainer
        )
        assert callback_trainer.model.conv1.out_channels == 64
        emit("explicit_server_trainer_callback", before_channels=32, after_channels=64)
    server = module.Server(
        model=model,
        algorithm=module.Algorithm,
        trainer=module.ServerTrainer,
        datasource=TinyDatasource,
    )
    server.callback_handler.add_callback(ServerObserver)
    original_response = server.customize_server_response
    original_aggregated = server.weights_aggregated

    def inspect_aggregate(updates: Any) -> None:
        actual = clone(server.require_algorithm().model.state_dict())
        error = compare(actual, server._retention_expected, parameters_only=True)
        server._retention_presort = actual
        emit(
            "aggregation_oracle",
            max_error=error,
            weighting="uniform channel average",
            parameter_tensors=sum("weight" in key or "bias" in key for key in actual),
        )
        if family == "anycostfl":
            import torch

            ties = []
            for key, value in actual.items():
                if value.ndim != 4 or "shortcut" in key:
                    continue
                independent = server._retention_expected[key]
                exact_norms = torch.linalg.vector_norm(value, dim=(1, 2, 3))
                independent_norms = torch.linalg.vector_norm(independent, dim=(1, 2, 3))
                if not torch.equal(
                    torch.argsort(exact_norms, descending=True),
                    torch.argsort(independent_norms, descending=True),
                ):
                    ordered = exact_norms.sort().values
                    ties.append(
                        {
                            "tensor": key,
                            "max_weight_error": float(
                                (value - independent).abs().max()
                            ),
                            "max_norm_error": float(
                                (exact_norms - independent_norms).abs().max()
                            ),
                            "min_adjacent_norm_gap": float(
                                (ordered[1:] - ordered[:-1]).min()
                            ),
                        }
                    )
            emit("aggregation_rounding_and_norm_ties", reordered_near_ties=ties)
        original_aggregated(updates)

    server.weights_aggregated = inspect_aggregate

    def select(response: dict[str, Any], client_id: int) -> dict[str, Any]:
        target = 0.5 if client_id == 1 else 1.0
        rates = server.require_algorithm().rates
        seed = next(
            candidate
            for candidate in range(1000)
            if float(random.Random(candidate).choice(rates)) == target
        )
        outside = random.getstate()
        random.seed(seed)
        state_hash = hashlib.sha256(pickle.dumps(random.getstate())).hexdigest()
        try:
            result = original_response(response, client_id)
        finally:
            random.setstate(outside)
        assert result["rate"] == target
        emit(
            "selected_rate",
            client_id=client_id,
            rate=result["rate"],
            seed=seed,
            rng_state_sha256=state_hash,
            limitation_activated=False,
        )
        return result

    server.customize_server_response = select
    original_start = mp.Process.start

    def observe_start(process: Any) -> None:
        original_start(process)
        emit(
            "child_started",
            child_pid=process.pid,
            created=psutil.Process(process.pid).create_time(),
        )

    cast(Any, mp.Process).start = observe_start
    cast(Any, base).run = observe_client_run
    client = module.create_client(
        model=model,
        datasource=TinyDatasource,
        callbacks=[ClientObserver],
        trainer_callbacks=[LossObserver],
    )
    server.run(client=client)


def resource_profile(model: Any) -> tuple[float, float]:
    import ptflops

    size = sys.getsizeof(pickle.dumps(model.state_dict())) / 1024**2
    macs, _ = ptflops.get_model_complexity_info(
        model, (3, 32, 32), as_strings=False, print_per_layer_stat=False, verbose=False
    )
    assert macs is not None
    flops = float(macs) / 1024**2
    assert math.isfinite(size) and math.isfinite(flops) and size > 0 and flops > 0
    return size, flops


def activated_budget(module: Any) -> None:
    import ptflops

    from plato.config import Config

    family = os.environ["PLATO_RETENTION_FAMILY"]
    model = module.resnet.resnet18 if family == "heterofl" else module.resnet18
    trainer = module.ServerTrainer(model=model)
    algorithm = module.Algorithm(trainer=trainer)
    records: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    original_profile = ptflops.get_model_complexity_info

    def observe_profile(model: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_profile(model, *args, **kwargs)
        queries.append({"model_rate": model.scaler.rate, "macs": float(result[0])})
        return result

    ptflops.get_model_complexity_info = observe_profile
    for candidate_rate in (1.0, 0.75 if family != "heterofl" else 0.5):
        candidate = model(
            model_rate=candidate_rate, **Config().parameters.client_model._asdict()
        )
        measured = resource_profile(candidate)
        budget = tuple(value * 1.03 for value in measured)
        selected = algorithm.choose_rate(budget, model)
        assert 0.5 <= selected <= 1.0
        if family != "heterofl":
            assert selected not in (0.5, 1.0)
        else:
            assert selected in algorithm.rates
        payload = algorithm.extract_weights()
        selected_model = model(
            model_rate=selected, **Config().parameters.client_model._asdict()
        )
        selected_model.load_state_dict(payload)
        actual = resource_profile(selected_model)
        assert all(
            value <= ceiling for value, ceiling in zip(actual, budget, strict=True)
        )
        loss = train_batch(selected_model)
        records.append(
            {
                "candidate_rate": candidate_rate,
                "profile": measured,
                "margin": 1.03,
                "budget": budget,
                "selected_rate": selected,
                "selected_profile": actual,
                "loss": loss,
            }
        )
    assert records[1]["selected_rate"] < records[0]["selected_rate"]
    emit(
        "activated_budget", limitation_activated=True, records=records, queries=queries
    )
    emit("success")


def dlg_update(module: Any) -> None:
    import torch

    from plato.config import Config

    factory = module.get()
    assert factory is not None
    model = factory()
    inputs, labels = fixture(2).tensors
    outputs = model(inputs)
    feature_outputs, features = model.forward_feature(inputs)
    assert outputs.shape == (2, 100) and features.shape[0] == 2
    torch.testing.assert_close(outputs, feature_outputs)
    loss = train_batch(model)
    emit(
        "dlg_update",
        model_name=Config().trainer.model_name,
        datasource_overlay=Config().data.datasource,
        output_shape=list(outputs.shape),
        feature_shape=list(features.shape),
        loss=loss,
    )
    emit("success")


def selector(module: Any) -> None:
    from plato.config import Config

    family = os.environ["PLATO_RETENTION_FAMILY"]
    retired = "mobilenet_v3_large" if family == "heterofl" else "vit"
    archive = {"heterofl": "heterofl-mobilenetv3", "dlg": "legacy-vit"}.get(
        family, family + "-vit"
    )
    for selected in (retired, "unknown_model"):
        Config().trainer = Config().trainer._replace(model_name=selected)
        if family == "dlg":
            Config().trainer = Config().trainer._replace(
                model_type="vit" if selected == retired else "unknown"
            )
        try:
            module.get() if family == "dlg" else module.main()
        except ValueError as error:
            message = str(error)
            if selected == retired:
                assert "retired" in message
                assert f"archives/retired/{archive}/README.md" in message
            else:
                assert "No such" in message and "unknown_model" in message
            emit("selector_diagnostic", selection=selected, message=message)
        else:
            raise AssertionError(f"Unsupported selector accepted: {selected}")
    emit("success")


def sysheterofl_round(module: Any) -> None:
    import torch

    from plato.config import Config

    factory = module.resnet.ResnetWrapper
    trainer = module.ServerTrainer(model=factory)
    algorithm = module.Algorithm(trainer=trainer)
    algorithm.initialize_arch_map(factory)
    assert len(algorithm.arch_list) == 2 and algorithm.max_loop == 1
    budget = tuple(
        max(row[index] for row in algorithm.size_flops_counts_dict) * 1.1
        for index in (0, 1)
    )
    # Preserve choose_config's actual greedy/exploration branch; seed the bounded run.
    random.seed(1234)
    selected = algorithm.choose_config(budget)
    payload = algorithm.extract_weights()
    subnet = factory(configs=selected, **Config().parameters.client_model._asdict())
    subnet.load_state_dict(payload)
    loss = train_batch(subnet)
    aggregate = algorithm.aggregation([subnet.state_dict()])
    algorithm.load_weights(aggregate)
    assert all(torch.isfinite(value).all() for value in aggregate.values())
    algorithm.model.eval()
    with torch.no_grad():
        assert torch.isfinite(algorithm.model(fixture(2).tensors[0])).all()
    emit(
        "sysheterofl_subnet",
        selected=selected,
        budget=budget,
        loss=loss,
        architecture_map=algorithm.arch_list,
        resources=algorithm.size_flops_counts_dict,
        max_loop=algorithm.max_loop,
        distillation_iterations=Config().parameters.distillation.iterations,
    )
    emit("success")


def main() -> None:
    from plato.config import Config

    Config()
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    config_path = Path(os.environ["config_file"])
    emit(
        "source_identity",
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        python=sys.version,
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    module = family_module()
    case = os.environ["PLATO_RETENTION_CASE"]
    if case == "socket":
        socket_round(module)
    elif case == "budget":
        activated_budget(module)
    elif case == "update":
        dlg_update(module)
    elif case == "selector":
        selector(module)
    elif case == "subnet":
        sysheterofl_round(module)
    elif case == "import":
        assert not any("archives/retired" in entry for entry in sys.path)
        emit("entrypoint_import", module_path=module.__file__, cwd=str(Path.cwd()))
        emit("success")
    else:
        raise ValueError(f"Unknown retention case: {case}")


if __name__ == "__main__":
    main()
