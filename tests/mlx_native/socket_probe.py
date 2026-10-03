"""Observe real MLX server/client synchronous entrypoints and socket codecs."""

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import psutil

from plato.callbacks.client import ClientCallback
from plato.callbacks.server import ServerCallback
from plato.clients import simple
from plato.config import Config
from plato.datasources.base import DataSource
from plato.models.mlx.lenet5 import LeNet5
from plato.samplers import registry as samplers
from plato.serialization.safetensor import serialize_tree
from plato.servers import base as server_base
from plato.servers import fedavg
from tests.mlx_native.helpers import dataset

ROOT = Path(os.environ["PLATO_MLX_ROOT"])
CASE = os.environ["PLATO_MLX_CASE"]


def emit(event, **values):
    with (ROOT / f"events-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(
            json.dumps(dict(event=event, pid=os.getpid(), wall=time.time(), **values))
            + "\n"
        )


def snapshot(event, weights, **values):
    filename = (
        f"{event}-{os.getpid()}-{values.get('client_id', 0)}"
        f"-round-{values.get('round', 0)}.safetensors"
    )
    (ROOT / filename).write_bytes(serialize_tree(weights))
    emit(event, weights_file=filename, **values)


class NativeData(DataSource):
    def __init__(self):
        super().__init__()
        self.trainset = self.testset = dataset(8, 1) + dataset(24, 2)


class NativePartition:
    def __init__(self, datasource, client_id, testing=False):
        self.indices = list(range(0, 8) if client_id == 1 else range(8, 32))

    def get(self):
        return self.indices

    def num_samples(self):
        return len(self.indices)


# Spawned workers import this module without executing the server's main.
samplers.registered_samplers["mlx_native_fixture"] = NativePartition


class ObserveClient(ClientCallback):
    def on_inbound_processed(self, client, data):
        snapshot(
            "client_baseline",
            data,
            client_id=client.client_id,
            round=client.current_round,
        )

    def on_outbound_ready(self, client, report, outbound_processor):
        snapshot(
            "client_trained",
            client.algorithm.extract_weights(),
            client_id=client.client_id,
            samples=report.num_samples,
            loss=client.trainer.context.state["last_loss"],
            device=str(client.trainer.context.device),
            round=client.current_round,
        )


class ObserveServer(ServerCallback):
    def on_training_will_start(self, server, **kwargs):
        snapshot("server_baseline", server.algorithm.extract_weights())

    def on_weights_received(self, server, weights_received):
        for update, weights in zip(server.updates, weights_received, strict=True):
            snapshot(
                "server_received",
                weights,
                client_id=update.client_id,
                samples=update.report.num_samples,
                round=server.current_round,
            )

    def on_weights_aggregated(self, server, updates):
        snapshot(
            "server_aggregate",
            server.algorithm.extract_weights(),
            round=server.current_round,
        )

    def on_clients_selected(self, server, selected_clients, **kwargs):
        emit(
            "round_started", round=server.current_round, clients=list(selected_clients)
        )

    def on_server_will_close(self, server, **kwargs):
        emit("server_close_started", round=server.current_round)


def observed_client_run(*args):
    from plato.client import run

    try:
        run(*args)
    except BaseException as error:
        emit("child_error", type=type(error).__name__, message=str(error))
        raise
    emit("child_returned", process_client_id=Config().args.id)


def main():
    original_start = mp.Process.start

    def observe_start(process):
        original_start(process)
        emit(
            "child_started",
            child_pid=process.pid,
            created=psutil.Process(process.pid).create_time(),
        )

    mp.Process.start = observe_start
    server_base.run = observed_client_run
    if CASE == "socket_native":
        client = simple.Client(
            model=LeNet5, datasource=NativeData, callbacks=[ObserveClient]
        )
        server = fedavg.Server(model=LeNet5, callbacks=[ObserveServer])
    elif CASE == "mnist_cached_simulation":
        # Use real model/trainer/datasource registries from the documented config.
        client = simple.Client(callbacks=[ObserveClient])
        server = fedavg.Server(callbacks=[ObserveServer])
    else:
        raise ValueError(CASE)
    server.run(client=client)
    emit("main_returned")


if __name__ == "__main__":
    from tests.mlx_native.socket_probe import main as guarded_main

    try:
        guarded_main()
    except BaseException as error:
        emit("caller_error", type=type(error).__name__, message=str(error))
        raise
