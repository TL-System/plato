"""Offline observations around the unchanged shipped FedDyn main/run wiring."""

import copy
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import TensorDataset

from plato.callbacks.server import ServerCallback
from plato.config import Config
from plato.datasources.base import DataSource
from plato.models.lenet5 import Model as LeNet
from plato.samplers import registry
from plato.servers import base as server_base
from plato.trainers.composable import ComposableTrainer

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples/customized_client_training/feddyn"
)
sys.path.insert(0, str(EXAMPLE))
import feddyn  # noqa: E402
import feddyn_client  # noqa: E402
import feddyn_server  # noqa: E402
import feddyn_trainer  # noqa: E402

ROOT = Path(os.environ["PLATO_FEDDYN_ROOT"])
KIND = os.environ["PLATO_FEDDYN_KIND"]


def emit(event, **values):
    with (ROOT / f"events-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(event=event, pid=os.getpid(), **values)) + "\n")


class Scalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.double))

    def forward(self, inputs):
        return self.theta.expand_as(inputs)


def quadratic(output, target):
    return (output - target).square().mean() / 2


class QuadraticTrainer(feddyn_trainer.Trainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.loss_strategy.base_loss_fn = quadratic


class Synthetic(DataSource):
    def __init__(self):
        super().__init__()
        n = Config().clients.total_clients
        if KIND == "lenet":
            inputs = (
                torch.arange(8 * 784, dtype=torch.float32).reshape(8, 1, 28, 28) / 6272
            )
            labels = torch.arange(8) % 10
        else:
            inputs = torch.ones(sum(range(1, n + 1)), 1, dtype=torch.double)
            labels = torch.cat(
                [torch.full((i, 1), i, dtype=torch.double) for i in range(1, n + 1)]
            )
        self.trainset = self.testset = TensorDataset(inputs, labels)


class Partition:
    def __init__(self, datasource, client_id, testing=False):
        count = (2 if client_id == 1 else 6) if KIND == "lenet" else client_id
        offset = (
            (0 if client_id == 1 else 2)
            if KIND == "lenet"
            else sum(range(1, client_id))
        )
        self.indices = list(range(offset, offset + count))

    def get(self):
        return self.indices

    def num_samples(self):
        return len(self.indices)


registry.registered_samplers["feddyn_fixture"] = Partition


class Observe(ServerCallback):
    def on_weights_received(self, server, weights):
        payloads = copy.deepcopy(weights)
        for payload in payloads:
            payload[1] = {
                key: value.item()
                if isinstance(value, np.ndarray) and value.shape == ()
                else value
                for key, value in payload[1].items()
            }
        torch.save(
            dict(
                payloads=payloads,
                before=copy.deepcopy(server.histories),
                baseline=copy.deepcopy(server.trainer.model.state_dict()),
                selected=list(server.selected_clients),
            ),
            ROOT / f"received-{server.current_round}.pth",
        )

    def on_weights_aggregated(self, server, updates):
        # Distinct global streams exercise the complete bundle, including cached
        # NumPy/Python Gaussian values. Selection uses its separate owned state.
        random.random()
        random.gauss(0, 1)
        np.random.standard_normal()
        torch.rand(3)

    def on_clients_processed(self, server):
        server.save_to_checkpoint()
        torch.save(
            server._committed_snapshot, ROOT / f"committed-{server.committed_round}.pth"
        )
        emit(
            "committed",
            round=server.committed_round,
            selected=list(server.selected_clients),
        )

    def on_server_will_close(self, server, **kwargs):
        emit("server_will_close", round=server.committed_round)


def observe_client_run(*args):
    from plato.client import run

    run(*args)
    emit("child_returned", client_id=Config().args.id)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    np.random.seed(29)
    random.seed(37)
    original_factory, original_server = (
        feddyn_client.create_client,
        feddyn_server.Server,
    )
    original_start, original_process = server_base.Server.start, mp.Process.start

    def client_factory():
        emit("custom_client")
        return original_factory(
            model=LeNet if KIND == "lenet" else Scalar,
            datasource=Synthetic,
            trainer=feddyn_trainer.Trainer if KIND == "lenet" else QuadraticTrainer,
        )

    def server_factory():
        emit("custom_server")
        return original_server(
            model=LeNet if KIND == "lenet" else Scalar,
            trainer=ComposableTrainer,
            callbacks=[Observe],
        )

    def start(server, *args, **kwargs):
        # Observation at entry to the REAL inherited start. Its registration,
        # listener and sockets execute normally after recording this snapshot.
        torch.save(
            dict(
                rng=server._rng_snapshot(),
                pending=server._pending_resume_rng,
                round=server.committed_round,
                run_id=server.run_id,
            ),
            ROOT / "start.pth",
        )
        return original_start(server, *args, **kwargs)

    def process_start(process):
        original_process(process)
        emit(
            "child_started",
            child_pid=process.pid,
            created=psutil.Process(process.pid).create_time(),
        )

    feddyn_client.create_client, feddyn_server.Server = client_factory, server_factory
    server_base.Server.start, server_base.run, mp.Process.start = (
        start,
        observe_client_run,
        process_start,
    )
    feddyn.main()
    emit("main_returned")


if __name__ == "__main__":
    from tests.integration.feddyn_entrypoint import main as guarded_main

    guarded_main()
