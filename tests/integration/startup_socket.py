"""Two independently trained CPU clients over Plato's real socket transport."""

import asyncio
import multiprocessing as mp
import time

import psutil
import torch
from torch import nn
from torch.utils.data import TensorDataset

from plato.callbacks.client import ClientCallback
from plato.callbacks.server import ServerCallback
from plato.clients import simple
from plato.config import Config
from plato.datasources import base as datasource_base
from plato.servers import base as server_base
from plato.servers import fedavg
from tests.integration.startup_probe import CASE, SentinelError, emit


def snapshot(weights):
    """Clone before later model loads; JSON has no references to live tensors."""
    return {
        name: value.detach().cpu().clone().tolist() for name, value in weights.items()
    }


class TinyModel(nn.Module):
    """A float-only classifier with the same fixed initialization in each process."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.2, -0.1], [-0.3, 0.4]]))
            self.linear.bias.zero_()

    def forward(self, inputs):
        return self.linear(inputs)


class TinyDatasource(datasource_base.DataSource):
    """Distinct deterministic local partitions, with realized counts 2 and 6."""

    def __init__(self):
        super().__init__()
        client_id = Config().args.id
        count = 2 if client_id == 1 else 6
        positions = torch.arange(1, count + 1, dtype=torch.float32)
        inputs = torch.stack((positions / count, positions.flip(0) / count), dim=1)
        inputs[:, 0] += client_id * 0.2
        targets = (torch.arange(count) + client_id) % 2
        self.trainset = TensorDataset(inputs, targets)
        self.testset = self.trainset
        emit(
            "local_data",
            process_client_id=client_id,
            count=count,
            inputs=inputs.tolist(),
        )


class ObserveClient(ClientCallback):
    def on_inbound_processed(self, client, data):
        emit("client_baseline", client_id=client.client_id, weights=snapshot(data))

    def on_outbound_ready(self, client, report, outbound_processor):
        emit(
            "client_trained",
            client_id=client.client_id,
            samples=report.num_samples,
            device=str(client.trainer.device),
            weights=snapshot(client.algorithm.extract_weights()),
        )
        if CASE == "socket_stall" and client.client_id == 2:
            emit("client_stall_entered", client_id=2)
            while True:
                time.sleep(1)


class ObserveServer(ServerCallback):
    def on_training_will_start(self, server, **kwargs):
        emit("server_baseline", weights=snapshot(server.algorithm.extract_weights()))

    def on_weights_received(self, server, weights_received):
        for update, weights in zip(server.updates, weights_received, strict=True):
            emit(
                "server_received",
                client_id=update.client_id,
                samples=update.report.num_samples,
                weights=snapshot(weights),
            )

    def on_weights_aggregated(self, server, updates):
        emit("server_aggregate", weights=snapshot(server.algorithm.extract_weights()))

    def on_server_will_close(self, server, **kwargs):
        emit("server_close_started", round=server.current_round)


class FailingServer(fedavg.Server):
    async def _periodic(self, periodic_interval):
        await asyncio.sleep(0.2)
        emit("original_failure", message="post-launch-sentinel")
        raise SentinelError("post-launch-sentinel")


class ObserveStalledServer(fedavg.Server):
    async def _client_payload_done(self, sid, client_id, s3_key=None):
        await super()._client_payload_done(sid, client_id, s3_key=s3_key)
        emit(
            "server_payload_processed",
            client_id=client_id,
            samples=self.reports[sid].num_samples,
            weights=snapshot(self.client_payload[sid]),
        )


def observe_client_run(*args):
    """Observe the real entrypoint's natural return in each spawned child."""
    from plato.client import run

    try:
        run(*args)
    except BaseException as error:
        emit("child_error", type=type(error).__name__, message=str(error))
        raise
    emit("child_returned", process_client_id=Config().args.id)


def real_round():
    """Keep spawning, training, processors, sockets and aggregation unmodified."""
    torch.set_num_threads(1)
    torch.manual_seed(7)
    original_start = mp.Process.start

    def observe_start(process):
        original_start(process)
        emit(
            "child_started",
            child_pid=process.pid,
            created=psutil.Process(process.pid).create_time(),
        )

    setattr(mp.Process, "start", observe_start)
    setattr(server_base, "run", observe_client_run)
    client = simple.Client(
        model=TinyModel, datasource=TinyDatasource, callbacks=[ObserveClient]
    )
    server_class = (
        FailingServer
        if CASE == "socket_failure"
        else ObserveStalledServer
        if CASE == "socket_stall"
        else fedavg.Server
    )
    server = server_class(model=TinyModel, callbacks=[ObserveServer])
    server.run(client=client)
