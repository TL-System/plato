"""Bounded synchronized E8 evidence; run only in an agreed idle host window."""

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROCESS_START = time.perf_counter()


def worker(root: Path, batch_size: int, number: int, repetition: int):
    import mlx.core as mx
    import numpy as np
    import psutil

    from plato.algorithms.mlx_fedavg import Algorithm
    from plato.config import Config
    from plato.models.mlx.lenet5 import LeNet5
    from plato.processors.safetensor_decode import Processor as Decode
    from plato.processors.safetensor_encode import Processor as Encode
    from plato.servers.strategies.aggregation import FedAvgAggregationStrategy
    from plato.servers.strategies.base import ServerContext
    from plato.trainers.mlx import ComposableMLXTrainer
    from tests.mlx_native.helpers import dataset, native_config, no_device_flags, update

    timings = {}

    def memory():
        return dict(
            active=mx.get_active_memory(),
            peak=mx.get_peak_memory(),
            cache=mx.get_cache_memory(),
            rss=psutil.Process().memory_info().rss,
            rss_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )

    def timed(name, action):
        mx.synchronize(mx.default_stream(mx.gpu))
        start = time.perf_counter()
        value = action()
        mx.synchronize(mx.default_stream(mx.gpu))
        timings[name] = time.perf_counter() - start
        return value

    with native_config(
        root / f"runtime-{number}",
        optimizer="adam",
        model_seed=17,
        training_seed=29,
        batch_size=batch_size,
    ):
        no_device_flags()
        trainer = ComposableMLXTrainer(model=LeNet5)
        trainer.set_client_id(number + 1)
        algorithm = Algorithm(trainer)
        samples = dataset(256, seed=41)
        baseline = algorithm.extract_weights()
        config = Config().trainer._asdict()
        timed("cold_local_train", lambda: trainer.train_model(config, samples, None))
        converted: tuple[mx.array, mx.array] = timed(
            "data_conversion",
            lambda: (
                mx.array(np.stack([x for x, _ in samples[:batch_size]])),
                mx.array(
                    np.array([y for _, y in samples[:batch_size]], dtype=np.int32)
                ),
            ),
        )
        optimizer = trainer.optimizer_strategy.create_optimizer(
            trainer.model, trainer.context
        )
        step = trainer.training_step_strategy

        def training_step():
            return step.training_step(
                trainer.model,
                optimizer,
                *converted,
                lambda outputs, labels: trainer.loss_strategy.compute_loss(
                    outputs, labels, trainer.context
                ),
                trainer.context,
            )

        for _ in range(10):
            training_step()
        mx.synchronize(trainer.context.stream)
        mx.reset_peak_memory()
        (root / f"ready-{number}").write_text(str(os.getpid()))
        deadline = time.monotonic() + 60
        while not (root / "go").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("benchmark launch barrier")
            time.sleep(0.01)
        measurements = []
        interval_start = time.time()
        for _ in range(100):
            mx.synchronize(trainer.context.stream)
            started = time.perf_counter()
            training_step()
            mx.synchronize(trainer.context.stream)
            measurements.append(time.perf_counter() - started)
        interval_end = time.time()
        after_steps = memory()
        weights = timed("owned_extraction", algorithm.extract_weights)
        encoded = timed("encode", lambda: Encode().process(weights))
        decoded = timed("decode", lambda: Decode().process(encoded))
        context = ServerContext()
        context.algorithm = algorithm
        context.trainer = trainer
        updates = [update(1, 8, decoded), update(2, 24, weights)]
        aggregate = timed(
            "aggregate",
            lambda: asyncio.run(
                FedAvgAggregationStrategy().aggregate_weights(
                    updates, baseline, [decoded, weights], context
                )
            ),
        )
        algorithm.validate_weights(aggregate, baseline)
        timed("checkpoint_save", lambda: trainer.save_model("bench.safetensors"))
        timed("checkpoint_load", lambda: trainer.load_model("bench.safetensors"))
        retention = []
        for round_id in range(1, 11):
            trainer.current_round = round_id
            timed(
                f"warm_local_round_{round_id}",
                lambda: trainer.train_model(config, samples, None),
            )
            current = algorithm.extract_weights()
            decoded_round = Decode().process(Encode().process(current))
            trainer._apply_model_state(decoded_round)
            del current, decoded_round
            gc.collect()
            mx.synchronize(trainer.context.stream)
            retention.append(dict(round=round_id, **memory()))
        result = dict(
            batch_size=batch_size,
            repetition=repetition,
            worker=number,
            pid=os.getpid(),
            device=str(trainer.context.device),
            stream=str(trainer.context.stream),
            architecture="native MLX LeNet5",
            optimizer="Adam",
            model_seed=17,
            training_seed=29,
            data_seed=41,
            warmup_steps=10,
            measured_steps=100,
            step_seconds=measurements,
            median=statistics.median(measurements),
            p95=float(np.percentile(measurements, 95)),
            interval=[interval_start, interval_end],
            after_steps=after_steps,
            timings=timings,
            retention=retention,
            payload_bytes=len(encoded),
            physical_workers=2
            if number >= 0 and root.name.startswith("concurrent")
            else 1,
            logical_samples=len(samples),
            local_max_concurrency=None,
            local_training_inline=True,
            local_comm_simulation=None,
            process_seconds=time.perf_counter() - PROCESS_START,
            limits="Local timed segments exclude sockets; actual fresh socket full rounds are separate.",
        )
        (root / f"worker-{number}.json").write_text(json.dumps(result, indent=2))


def driver(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for simultaneous in (False, True):
        for batch_size in (8, 32, 64):
            for repetition in range(3):
                name = f"{'concurrent' if simultaneous else 'isolated'}-{batch_size}-{repetition}"
                run = root / name
                run.mkdir()
                processes = []
                files = []
                started = time.monotonic()
                for number in range(2 if simultaneous else 1):
                    command = [
                        sys.executable,
                        "-B",
                        "-m",
                        "tests.mlx_native.benchmark",
                        "--worker",
                        "--root",
                        str(run),
                        "--batch",
                        str(batch_size),
                        "--number",
                        str(number),
                        "--repetition",
                        str(repetition),
                    ]
                    stream = (run / f"worker-{number}.log").open("w")
                    files.append(stream)
                    processes.append(
                        subprocess.Popen(
                            command,
                            stdout=stream,
                            stderr=stream,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                        )
                    )
                try:
                    deadline = time.monotonic() + 60
                    while not all(
                        (run / f"ready-{i}").exists() for i in range(len(processes))
                    ):
                        if any(p.poll() is not None for p in processes):
                            raise RuntimeError(f"benchmark worker failed; see {run}")
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"benchmark readiness; see {run}")
                        time.sleep(0.01)
                    (run / "go").write_text("start")
                    for process in processes:
                        assert process.wait(timeout=90) == 0, run
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
                    for stream in files:
                        stream.close()
                workers = [
                    json.loads((run / f"worker-{i}.json").read_text())
                    for i in range(len(processes))
                ]
                overlap = min(w["interval"][1] for w in workers) - max(
                    w["interval"][0] for w in workers
                )
                if simultaneous:
                    assert overlap > 0, "clients did not execute simultaneously"
                results.append(
                    dict(
                        name=name,
                        elapsed=time.monotonic() - started,
                        simultaneous=simultaneous,
                        overlap=overlap,
                        workers=workers,
                    )
                )
    from plato.serialization.safetensor import deserialize_tree
    from tests.mlx_native.helpers import assert_tree_equal
    from tests.mlx_native.socket_harness import run_probe
    from tests.mlx_native.test_phase3_runtime import paired_reference

    socket_rounds = []
    for batch_size in (8, 32, 64):
        for repetition in range(3):
            run = root / f"socket-{batch_size}-{repetition}"
            result = run_probe(
                run, "socket_native", timeout=120, batch_size=batch_size, rounds=3
            )
            assert result["returncode"] == 0 and not result["timed_out"]
            assert not result["forced"] and not result["survivors"]
            events = result["events"]
            assert len([e for e in events if e["event"] == "server_aggregate"]) == 3
            round_seconds = []
            for round_id in range(1, 4):
                start = next(
                    e["wall"]
                    for e in events
                    if e["event"] == "round_started" and e["round"] == round_id
                )
                finish = next(
                    e["wall"]
                    for e in events
                    if e["event"] == "server_aggregate" and e["round"] == round_id
                )
                round_seconds.append(finish - start)
                arrivals = sorted(
                    [
                        e
                        for e in events
                        if e["event"] == "server_received" and e["round"] == round_id
                    ],
                    key=lambda e: e["client_id"],
                )
                assert [e["samples"] for e in arrivals] == [8, 24]

                def weights(event):
                    return deserialize_tree((run / event["weights_file"]).read_bytes())

                actual = weights(
                    next(
                        e
                        for e in events
                        if e["event"] == "server_aggregate" and e["round"] == round_id
                    )
                )
                expected = paired_reference(*(weights(e) for e in arrivals))
                assert_tree_equal(actual, expected, rtol=1e-5, atol=1e-6)
            socket_rounds.append(
                dict(
                    batch_size=batch_size,
                    repetition=repetition,
                    elapsed=result["elapsed"],
                    physical_workers=2,
                    logical_clients=2,
                    samples=[8, 24],
                    max_concurrency=2,
                    comm_simulation=False,
                    simulate_wall_time=False,
                    receipt=str(run / "result.json"),
                    cold_round=round_seconds[0],
                    warm_rounds=round_seconds[1:],
                )
            )
    receipt = dict(
        host=platform.platform(),
        machine=platform.machine(),
        interpreter=sys.version,
        executable=sys.executable,
        versions={
            n: importlib.metadata.version(n)
            for n in ("mlx", "mlx-metal", "numpy", "torch", "torchvision")
        },
        numerical=results,
        socket_rounds=socket_rounds,
        eager_default=True,
        compilation_adopted=False,
        comparison="No stock PyTorch architecture or unqualified compiled comparison.",
    )
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--number", type=int, default=0)
    parser.add_argument("--repetition", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments.root, arguments.batch, arguments.number, arguments.repetition)
    else:
        driver(arguments.root)
