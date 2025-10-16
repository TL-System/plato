"""
The base class for federated learning servers.
"""

import asyncio
import heapq
import logging
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from abc import abstractmethod
from types import SimpleNamespace

import numpy as np
import socketio
from aiohttp import web

from plato.callbacks.handler import CallbackHandler
from plato.callbacks.server import LogProgressCallback
from plato.client import run
from plato.config import Config
from plato.servers.strategies.base import ServerContext
from plato.servers.strategies.client_selection import RandomSelectionStrategy
from plato.utils import fonts, s3


# pylint: disable=unused-argument, protected-access
class ServerEvents(socketio.AsyncNamespace):
    """A custom namespace for socketio.AsyncServer."""

    def __init__(self, namespace, plato_server):
        super().__init__(namespace)
        self.plato_server = plato_server

    async def on_connect(self, sid, environ):
        """Upon a new connection from a client."""
        logging.info("[Server #%d] A new client just connected.", os.getpid())

    async def on_disconnect(self, sid, reason=None):
        """Upon a disconnection event."""
        logging.info("[Server #%d] An existing client just disconnected.", os.getpid())
        await self.plato_server._client_disconnected(sid)

    async def on_client_alive(self, sid, data):
        """A new client arrived."""
        await self.plato_server.register_client(sid, data["pid"], data["id"])

    async def on_client_report(self, sid, data):
        """An existing client sends a new report from local training."""
        await self.plato_server._client_report_arrived(sid, data["id"], data["report"])

    async def on_chunk(self, sid, data):
        """A chunk of data from the server arrived."""
        await self.plato_server._client_chunk_arrived(sid, data["data"])

    async def on_client_payload(self, sid, data):
        """An existing client sends a new payload from local training."""
        await self.plato_server._client_payload_arrived(sid, data["id"])

    async def on_client_payload_done(self, sid, data):
        """An existing client finished sending its payloads from local training."""
        if "s3_key" in data:
            await self.plato_server._client_payload_done(
                sid, data["id"], s3_key=data["s3_key"]
            )
        else:
            await self.plato_server._client_payload_done(sid, data["id"])


class Server:
    """The base class for federated learning servers."""

    def __init__(self, callbacks=None, client_selection_strategy=None):
        self.sio = None
        self.client = None
        self.clients = {}
        self.total_clients = 0
        # The client ids are stored for client selection
        self.clients_pool = []
        self.clients_per_round = 0
        self.selected_clients = None
        self.selected_client_id = 0
        self.selected_sids = []
        self.current_round = 0
        self.resumed_session = False
        self.algorithm = None
        self.trainer = None
        self.accuracy = 0
        self.accuracy_std = 0
        self.reports = {}
        self.updates = []
        self.client_payload = {}
        self.client_chunks = {}
        self.s3_client = None
        self.outbound_processor = None
        self.inbound_processor = None
        self.comm_simulation = (
            Config().clients.comm_simulation
            if hasattr(Config().clients, "comm_simulation")
            else True
        )

        self._mpc_round_lock = None
        if getattr(Config().clients, "type", None) == "mpc" and not hasattr(
            Config().server, "s3_endpoint_url"
        ):
            spawn_context = mp.get_context("spawn")
            self._mpc_round_lock = spawn_context.Lock()

        # Starting from the default server callback class, add all supplied server callbacks
        self.callbacks = [LogProgressCallback]
        if callbacks is not None:
            self.callbacks.extend(callbacks)
        self.callback_handler = CallbackHandler(self.callbacks)

        # Initialize server context for strategies
        self.context = ServerContext()

        # Initialize client selection strategy (default: random selection)
        self.client_selection_strategy = (
            client_selection_strategy or RandomSelectionStrategy()
        )

        # Accumulated communication overhead (MB) throughout the FL training session
        self.comm_overhead = 0

        # Downlink and uplink bandwidth (Mbps)
        # for computing communication time in communication simulation mode
        self.downlink_bandwidth = (
            Config().server.downlink_bandwidth
            if hasattr(Config().server, "downlink_bandwidth")
            else 100
        )
        self.uplink_bandwidth = (
            Config().server.uplink_bandwidth
            if hasattr(Config().server, "uplink_bandwidth")
            else 100
        )
        if Config().is_edge_server():
            if hasattr(Config().server, "edge_downlink_bandwidth"):
                self.downlink_bandwidth = Config().server.edge_downlink_bandwidth
            if hasattr(Config().server, "edge_uplink_bandwidth"):
                self.uplink_bandwidth = Config().server.edge_uplink_bandwidth

        # Use dictionaries to record downlink/uplink communication time of each client
        self.downlink_comm_time = {}
        self.uplink_comm_time = {}

        # States that need to be maintained for asynchronous FL

        # sids that are currently in use
        self.training_sids = []

        # Clients whose new reports were received but not yet processed
        self.reported_clients = []

        # Clients who are still training since the last round of aggregation
        self.training_clients = {}

        # The wall clock time that is simulated to accommodate the fact that
        # clients can only run a batch at a time, controlled by `max_concurrency`
        self.initial_wall_time = time.time()
        self.wall_time = time.time()

        # The wall clock time when a communication round starts
        self.round_start_wall_time = self.wall_time

        # When simulating the wall clock time, the server needs to remember the
        # set of reporting clients received since the previous round of aggregation
        self.current_reported_clients = {}
        self.current_processed_clients = {}
        self.prng_state = random.getstate()

        self.ping_interval = 3600
        self.ping_timeout = 3600
        self.asynchronous_mode = False
        self.periodic_interval = 5
        self.staleness_bound = 1000
        self.minimum_clients = 1
        self.simulate_wall_time = False
        self.request_update = False
        self.disable_clients = False

        # With specifying max_concurrency, selected clients run batch by batach
        # The number of clients in a batch on an available device is the same as the max_concurrency
        # This list contains ids of selected clients that has run in the current round
        if hasattr(Config().trainer, "max_concurrency"):
            self.trained_clients = []

    def __repr__(self):
        return f"Server #{os.getpid()}"

    def __str__(self):
        return f"Server #{os.getpid()}"

    def configure(self) -> None:
        """Initializes configuration settings based on the configuration file."""
        logging.info("[%s] Configuring the server...", self)

        # Ping interval and timeout setup for the server
        self.ping_interval = (
            Config().server.ping_interval
            if hasattr(Config().server, "ping_interval")
            else 3600
        )
        self.ping_timeout = (
            Config().server.ping_timeout
            if hasattr(Config().server, "ping_timeout")
            else 3600
        )

        # Are we operating in asynchronous mode?
        self.asynchronous_mode = (
            hasattr(Config().server, "synchronous") and not Config().server.synchronous
        )

        # What is the periodic interval for running our periodic task in asynchronous mode?
        self.periodic_interval = (
            Config().server.periodic_interval
            if hasattr(Config().server, "periodic_interval")
            else 5
        )

        # The staleness threshold is used to determine if a training clients should be
        # considered 'stale', if their starting round is too much behind the current round
        # on the server
        self.staleness_bound = (
            Config().server.staleness_bound
            if hasattr(Config().server, "staleness_bound")
            else 0
        )

        if not Config().is_central_server():
            # What is the minimum number of clients that must have reported before aggregation
            # takes place?
            self.minimum_clients = (
                Config().server.minimum_clients_aggregated
                if hasattr(Config().server, "minimum_clients_aggregated")
                else 1
            )
        else:
            # In cross-silo FL, what is the minimum number of edge servers that must have reported
            # before the central server conduct aggregation?
            self.minimum_clients = (
                Config().server.minimum_edges_aggregated
                if hasattr(Config().server, "minimum_edges_aggregated")
                else Config().algorithm.total_silos
            )

        # Are we simulating the wall clock time on the server? This is useful when the clients
        # are training in batches due to a lack of memory on the GPUs
        self.simulate_wall_time = (
            hasattr(Config().server, "simulate_wall_time")
            and Config().server.simulate_wall_time
        )

        # Do we wish to send urgent requests for model updates to the slow clients?
        self.request_update = (
            hasattr(Config().server, "request_update")
            and Config().server.request_update
        )

        # Are we disabling all clients and prevent them from running?
        self.disable_clients = (
            hasattr(Config().server, "disable_clients")
            and Config().server.disable_clients
        )

        # Compute the per-client uplink bandwidth
        if self.asynchronous_mode:
            self.uplink_bandwidth = self.uplink_bandwidth / self.minimum_clients
        else:
            self.uplink_bandwidth = self.uplink_bandwidth / self.clients_per_round

        # Setup server context for strategies
        self.context.server = self
        self.context.total_clients = self.total_clients
        self.context.clients_per_round = self.clients_per_round
        self.context.state["prng_state"] = self.prng_state

        # Setup client selection strategy
        self.client_selection_strategy.setup(self.context)

    def run(self, client=None, edge_server=None, edge_client=None, trainer=None):
        """Starts a run loop for the server."""
        self.client = client
        self.configure()

        if Config().args.resume:
            self._resume_from_checkpoint()

        client_kwargs = None
        if getattr(Config().clients, "type", None) == "mpc":
            client_kwargs = {
                "round_store_lock": self._mpc_round_lock,
                "debug_artifacts": getattr(
                    Config().clients, "mpc_debug_artifacts", False
                ),
            }

        if Config().is_central_server():
            # Start the edge servers as clients of the central server first
            # Once all edge servers are live, clients will be initialized in the
            # training_will_start() event call of the central server
            Server._start_clients(
                as_server=True,
                client=self.client,
                edge_server=edge_server,
                edge_client=edge_client,
                trainer=trainer,
                client_kwargs=client_kwargs,
            )

            asyncio.get_event_loop().create_task(self._periodic(self.periodic_interval))
            if hasattr(Config().server, "random_seed"):
                seed = Config().server.random_seed
                logging.info("Setting the random seed for selecting clients: %s", seed)
                random.seed(seed)
                self.prng_state = random.getstate()
            self.start()

        else:
            if self.disable_clients:
                logging.info("No clients are launched (server:disable_clients = true)")
            else:
                Server._start_clients(client=self.client, client_kwargs=client_kwargs)

            asyncio.get_event_loop().create_task(self._periodic(self.periodic_interval))

            if hasattr(Config().server, "random_seed"):
                seed = Config().server.random_seed
                logging.info("Setting the random seed for selecting clients: %s", seed)
                random.seed(seed)
                self.prng_state = random.getstate()

            self.start()

    def start(self, port=Config().server.port):
        """Starts running the socket.io server."""
        logging.info(
            "Starting a server at address %s and port %s.",
            Config().server.address,
            port,
        )

        self.sio = socketio.AsyncServer(
            ping_interval=self.ping_interval,
            max_http_buffer_size=2**31,
            ping_timeout=self.ping_timeout,
        )
        self.sio.register_namespace(ServerEvents(namespace="/", plato_server=self))

        if hasattr(Config().server, "s3_endpoint_url"):
            self.s3_client = s3.S3()

        app = web.Application()
        self.sio.attach(app)
        web.run_app(
            app,
            host=Config().server.address,
            port=port,
            loop=asyncio.get_event_loop(),
        )

    async def register_client(self, sid, client_process_id, client_id):
        """Adds a newly arrived client to the list of clients."""
        self.clients[client_process_id] = {
            "sid": sid,
            "client_id": client_id,
        }
        logging.info("[%s] New client with id #%d arrived.", self, client_id)
        logging.info("[%s] Client process #%d registered.", self, client_process_id)

        if (
            hasattr(Config().trainer, "max_concurrency")
            and not Config().is_central_server()
        ):
            required_launched_clients = min(
                Config().trainer.max_concurrency * max(1, Config().gpu_count()),
                self.clients_per_round,
            )
        else:
            required_launched_clients = self.clients_per_round

        if (self.current_round == 0 or self.resumed_session) and len(
            self.clients
        ) >= required_launched_clients:
            self.resumed_session = False

            self.training_will_start()
            self.callback_handler.call_event("on_training_will_start", self)

            await self._select_clients()

    @staticmethod
    def _start_clients(
        client=None,
        as_server=False,
        edge_server=None,
        edge_client=None,
        trainer=None,
        client_kwargs=None,
    ):
        """Starts all the clients as separate processes."""
        starting_id = 1

        # We only need to launch the number of clients necessary for concurrent training
        # If `max_concurrency` in `trainer` is specified, the limit number is
        # `max_concurrency` multiply the number of available devices
        # (multiply number of edge servers in cross-silo training)
        if hasattr(Config().trainer, "max_concurrency"):
            if Config().is_central_server():
                client_processes = min(
                    Config().trainer.max_concurrency
                    * max(1, Config().gpu_count())
                    * Config().algorithm.total_silos,
                    Config().clients.per_round,
                )
            else:
                client_processes = min(
                    Config().trainer.max_concurrency * max(1, Config().gpu_count()),
                    Config().clients.per_round,
                )
        # Otherwise, the limited number is the same as the number of clients per round
        else:
            client_processes = Config().clients.per_round

        if as_server:
            total_processes = Config().algorithm.total_silos
            starting_id += Config().clients.total_clients
        else:
            total_processes = client_processes

        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)

        for client_id in range(starting_id, total_processes + starting_id):
            if as_server:
                port = int(Config().server.port) + client_id
                logging.info(
                    "Starting client #%d as an edge server on port %s.",
                    client_id,
                    port,
                )
                proc = mp.Process(
                    target=run,
                    args=(
                        client_id,
                        port,
                        client,
                        edge_server,
                        edge_client,
                        trainer,
                        client_kwargs,
                    ),
                )
                proc.start()
            else:
                logging.info("Starting client #%d's process.", client_id)
                proc = mp.Process(
                    target=run,
                    args=(
                        client_id,
                        None,
                        client,
                        None,
                        None,
                        None,
                        client_kwargs,
                    ),
                )
                proc.start()

    async def _close_connections(self):
        """Closes all socket.io connections after training completes."""
        for client_id, client in dict(self.clients).items():
            logging.info("Closing the connection to client #%d.", client_id)
            await self.sio.emit("disconnect", room=client["sid"])

    async def _select_clients(self, for_next_batch=False):
        """Selects a subset of the clients and send messages to them to start training."""
        if not for_next_batch:
            self.updates = []
            self.current_round += 1
            self.round_start_wall_time = self.wall_time

            if hasattr(Config().trainer, "max_concurrency"):
                self.trained_clients = []

            logging.info(
                fonts.colourize(
                    f"\n[{self}] Starting round {self.current_round}/{Config().trainer.rounds}."
                )
            )

            if Config().is_central_server():
                # In cross-silo FL, the central server selects from the pool of edge servers
                self.clients_pool = list(self.clients)

            elif not Config().is_edge_server():
                self.clients_pool = list(range(1, 1 + self.total_clients))

            # In asychronous FL, avoid selecting new clients to replace those that are still
            # training at this time

            # When simulating the wall clock time, if len(self.reported_clients) is 0, the
            # server has aggregated all reporting clients already
            if (
                self.asynchronous_mode
                and self.selected_clients is not None
                and len(self.reported_clients) > 0
                and len(self.reported_clients) < self.clients_per_round
            ):
                # If self.selected_clients is None, it implies that it is the first iteration;
                # If len(self.reported_clients) == self.clients_per_round, it implies that
                # all selected clients have already reported.

                # Except for these two cases, we need to exclude the clients who are still
                # training.
                training_client_ids = [
                    self.training_clients[client_id]["id"]
                    for client_id in self.training_clients
                ]

                # If the server is simulating the wall clock time, some of the clients who
                # reported may not have been aggregated; they should be excluded from the next
                # round of client selection
                reporting_client_ids = [
                    client[2]["client_id"] for client in self.reported_clients
                ]

                selectable_clients = [
                    client
                    for client in self.clients_pool
                    if client not in training_client_ids
                    and client not in reporting_client_ids
                ]

                if self.simulate_wall_time:
                    self.selected_clients = self.choose_clients(
                        selectable_clients, len(self.current_processed_clients)
                    )
                else:
                    self.selected_clients = self.choose_clients(
                        selectable_clients, len(self.reported_clients)
                    )
            else:
                self.selected_clients = self.choose_clients(
                    self.clients_pool, self.clients_per_round
                )

            self.current_reported_clients = {}
            self.current_processed_clients = {}

            # There is no need to clear the list of reporting clients if we are
            # simulating the wall clock time on the server. This is because
            # when wall clock time is simulated, the server needs to wait for
            # all the clients to report before selecting a subset of clients for
            # replacement, and all remaining reporting clients will be processed
            # in the next round
            if not self.simulate_wall_time:
                self.reported_clients = []

        if len(self.selected_clients) > 0:
            self.selected_sids = []

            # If max_concurrency is specified, run selected clients batch by batch,
            # and the number of clients in each batch (on each GPU, if multiple GPUs are available)
            # is equal to # (or maybe smaller than for the last batch) max_concurrency
            if (
                hasattr(Config().trainer, "max_concurrency")
                and not Config().is_central_server()
            ):
                selected_clients = []
                if Config().gpu_count() > 1:
                    untrained_clients = list(
                        set(self.selected_clients).difference(self.trained_clients)
                    )
                    available_gpus = Config().gpu_count()
                    for cuda_id in range(available_gpus):
                        for client_id in untrained_clients:
                            if client_id % available_gpus == cuda_id:
                                selected_clients.append(client_id)
                            if len(selected_clients) >= min(
                                len(self.clients),
                                (cuda_id + 1) * Config().trainer.max_concurrency,
                                self.clients_per_round,
                            ):
                                break
                        # There is no enough alive clients, break the selection
                        if len(selected_clients) >= len(self.clients):
                            break
                else:
                    selected_clients = self.selected_clients[
                        len(self.trained_clients) : min(
                            len(self.trained_clients) + len(self.clients),
                            len(self.selected_clients),
                        )
                    ]

                self.trained_clients += selected_clients

            else:
                selected_clients = self.selected_clients

            for selected_client_id in selected_clients:
                self.selected_client_id = selected_client_id

                if Config().is_central_server():
                    client_process_id = selected_client_id
                else:
                    client_processes = [client for client in self.clients]

                    # Find a client process that is currently not training
                    # or selected in this round
                    for process_id in client_processes:
                        current_sid = self.clients[process_id]["sid"]
                        if not (
                            current_sid in self.training_sids
                            or current_sid in self.selected_sids
                        ):
                            client_process_id = process_id
                            break

                sid = self.clients[client_process_id]["sid"]

                # Track the selected client process
                self.training_sids.append(sid)
                self.selected_sids.append(sid)

                # Assign the client id to the client process
                self.clients[client_process_id]["client_id"] = self.selected_client_id

                self.training_clients[self.selected_client_id] = {
                    "id": self.selected_client_id,
                    "starting_round": self.current_round,
                    "start_time": self.round_start_wall_time,
                    "update_requested": False,
                }

                logging.info(
                    "[%s] Selecting client #%d for training.",
                    self,
                    self.selected_client_id,
                )

                server_response = {
                    "id": self.selected_client_id,
                    "current_round": self.current_round,
                }
                server_response = self.customize_server_response(
                    server_response, client_id=self.selected_client_id
                )

                payload = self.algorithm.extract_weights()
                payload = self.customize_server_payload(payload)

                if self.comm_simulation:
                    logging.info(
                        "[%s] Sending the current model to client #%d (simulated).",
                        self,
                        self.selected_client_id,
                    )

                    # First apply outbound processors, if any
                    payload = self.outbound_processor.process(payload)

                    model_name = (
                        Config().trainer.model_name
                        if hasattr(Config().trainer, "model_name")
                        else "custom"
                    )
                    if "/" in model_name:
                        model_name = model_name.replace("/", "_")

                    checkpoint_path = Config().params["checkpoint_path"]

                    payload_filename = (
                        f"{checkpoint_path}/{model_name}_{self.selected_client_id}.pth"
                    )

                    with open(payload_filename, "wb") as payload_file:
                        pickle.dump(payload, payload_file)

                    server_response["payload_filename"] = payload_filename

                    payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2

                    logging.info(
                        "[%s] Sending %.2f MB of payload data to client #%d (simulated).",
                        self,
                        payload_size,
                        self.selected_client_id,
                    )

                    self.comm_overhead += payload_size

                    # Compute the communication time to transfer the current global model to client
                    self.downlink_comm_time[self.selected_client_id] = payload_size / (
                        (self.downlink_bandwidth / 8) / len(self.selected_clients)
                    )

                # Send the server response as metadata to the clients (payload to follow)
                await self.sio.emit(
                    "payload_to_arrive", {"response": server_response}, room=sid
                )

                if not self.comm_simulation:
                    # Send the server payload to the client
                    logging.info(
                        "[%s] Sending the current model to client #%d.",
                        self,
                        selected_client_id,
                    )

                    await self._send(sid, payload, selected_client_id)

            self.clients_selected(self.selected_clients)
            self.callback_handler.call_event(
                "on_clients_selected", self, self.selected_clients
            )

    def choose_clients(self, clients_pool, clients_count):
        """Chooses a subset of clients to participate in each round."""
        assert clients_count <= len(clients_pool)

        if type(self).choose_clients is not Server.choose_clients:
            raise RuntimeError(
                "Custom choose_clients overrides should use the strategy API."
                " Please update the subclass to call `_select_clients_with_strategy`."
            )

        return self._select_clients_with_strategy(clients_pool, clients_count)

    def _select_clients_with_strategy(self, clients_pool, clients_count):
        """Delegate client selection to the configured strategy."""
        self.context.current_round = self.current_round
        self.context.state["prng_state"] = self.prng_state

        selected_clients = self.client_selection_strategy.select_clients(
            clients_pool, clients_count, self.context
        )

        # Update PRNG state from context
        self.prng_state = self.context.state["prng_state"]

        # Call strategy hook
        self.client_selection_strategy.on_clients_selected(
            selected_clients, self.context
        )

        return selected_clients

    async def _periodic(self, periodic_interval):
        """Runs _periodic_task() periodically on the server. The time interval between
        its execution is defined in 'server:periodic_interval'.
        """
        while True:
            await self._periodic_task()
            await asyncio.sleep(periodic_interval)

    async def _periodic_task(self):
        """A periodic task that is executed from time to time, determined by
        'server:periodic_interval' with a default value of 5 seconds, in the configuration.
        """
        # Call the async function that defines a customized periodic task, if any
        await self.periodic_task()

        # If we are operating in asynchronous mode, aggregate the model updates received so far.
        if self.asynchronous_mode and not self.simulate_wall_time:
            # Is there any training clients who are currently training on models that are too
            # `stale,` as defined by the staleness threshold?
            for __, client_data in self.training_clients.items():
                # The client is still working at an early round, early enough to stop the
                # aggregation process as determined by 'staleness'
                client_staleness = self.current_round - client_data["starting_round"]
                if client_staleness > self.staleness_bound:
                    logging.info(
                        "[%s] Client %s is still working at round %s, which is "
                        "beyond the staleness bound %s compared to the current round %s. "
                        "Nothing to process.",
                        self,
                        client_data["id"],
                        client_data["starting_round"],
                        self.staleness_bound,
                        self.current_round,
                    )

                    return

            if len(self.updates) >= self.minimum_clients:
                logging.info(
                    "[%s] %d client report(s) received in asynchronous mode. Processing.",
                    self,
                    len(self.updates),
                )
                await self._process_reports()
                await self.wrap_up()
                await self._select_clients()
            else:
                logging.info(
                    "[%s] No sufficient number of client reports have been received. "
                    "Nothing to process.",
                    self,
                )

    async def _send_in_chunks(self, data, sid, client_id) -> None:
        """Sends a bytes object in fixed-sized chunks to the client."""
        step = 1024**2
        chunks = [data[i : i + step] for i in range(0, len(data), step)]

        for chunk in chunks:
            await self.sio.emit("chunk", {"data": chunk}, room=sid)

        await self.sio.emit("payload", {"id": client_id}, room=sid)

    async def _send(self, sid, payload, client_id) -> None:
        """Sends a new data payload to the client using either S3 or socket.io."""
        # First apply outbound processors, if any
        payload = self.outbound_processor.process(payload)

        metadata = {"id": client_id}

        if self.s3_client is not None:
            s3_key = f"server_payload_{os.getpid()}_{self.current_round}"
            self.s3_client.send_to_s3(s3_key, payload)
            data_size = sys.getsizeof(pickle.dumps(payload))
            metadata["s3_key"] = s3_key
        else:
            data_size = 0

            if isinstance(payload, list):
                for data in payload:
                    _data = pickle.dumps(data)
                    await self._send_in_chunks(_data, sid, client_id)
                    data_size += sys.getsizeof(_data)

            else:
                _data = pickle.dumps(payload)
                await self._send_in_chunks(_data, sid, client_id)
                data_size = sys.getsizeof(_data)

        await self.sio.emit("payload_done", metadata, room=sid)

        logging.info(
            "[%s] Sent %.2f MB of payload data to client #%d.",
            self,
            data_size / 1024**2,
            client_id,
        )

        self.comm_overhead += data_size / 1024**2

    async def _client_report_arrived(self, sid, client_id, report):
        """Upon receiving a report from a client."""
        self.reports[sid] = pickle.loads(report)
        self.client_payload[sid] = None
        self.client_chunks[sid] = []

        if self.comm_simulation:
            model_name = (
                Config().trainer.model_name
                if hasattr(Config().trainer, "model_name")
                else "custom"
            )
            if "/" in model_name:
                model_name = model_name.replace("/", "_")
            checkpoint_path = Config().params["checkpoint_path"]
            payload_filename = f"{checkpoint_path}/{model_name}_client_{client_id}.pth"
            with open(payload_filename, "rb") as payload_file:
                self.client_payload[sid] = pickle.load(payload_file)

            payload_size = (
                sys.getsizeof(pickle.dumps(self.client_payload[sid])) / 1024**2
            )

            logging.info(
                "[%s] Received %.2f MB of payload data from client #%d (simulated).",
                self,
                payload_size,
                client_id,
            )

            self.comm_overhead += payload_size

            self.uplink_comm_time[client_id] = payload_size / (
                self.uplink_bandwidth / 8
            )

            await self.process_client_info(client_id, sid)

    async def _client_chunk_arrived(self, sid, data) -> None:
        """Upon receiving a chunk of data from a client."""
        self.client_chunks[sid].append(data)

    async def _client_payload_arrived(self, sid, client_id):
        """Upon receiving a portion of the payload from a client."""
        assert len(self.client_chunks[sid]) > 0 and client_id in self.training_clients

        payload = b"".join(self.client_chunks[sid])
        _data = pickle.loads(payload)
        self.client_chunks[sid] = []

        if self.client_payload[sid] is None:
            self.client_payload[sid] = _data
        elif isinstance(self.client_payload[sid], list):
            self.client_payload[sid].append(_data)
        else:
            self.client_payload[sid] = [self.client_payload[sid]]
            self.client_payload[sid].append(_data)

    async def _client_payload_done(self, sid, client_id, s3_key=None):
        """Upon receiving all the payload from a client, either via S3 or socket.io."""
        if s3_key is None:
            assert self.client_payload[sid] is not None

            payload_size = 0
            if isinstance(self.client_payload[sid], list):
                for _data in self.client_payload[sid]:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            else:
                payload_size = sys.getsizeof(pickle.dumps(self.client_payload[sid]))
        else:
            self.client_payload[sid] = self.s3_client.receive_from_s3(s3_key)
            payload_size = sys.getsizeof(pickle.dumps(self.client_payload[sid]))

        logging.info(
            "[%s] Received %.2f MB of payload data from client #%d.",
            self,
            payload_size / 1024**2,
            client_id,
        )

        self.comm_overhead += payload_size / 1024**2

        await self.process_client_info(client_id, sid)

    async def process_client_info(self, client_id, sid):
        """Processes the received metadata information from a reporting client."""
        # First pass through the inbound_processor(s), if any
        self.client_payload[sid] = self.inbound_processor.process(
            self.client_payload[sid]
        )

        if self.comm_simulation:
            if (
                hasattr(Config().clients, "compute_comm_time")
                and Config().clients.compute_comm_time
            ):
                self.reports[sid].comm_time = (
                    self.downlink_comm_time[client_id]
                    + self.uplink_comm_time[client_id]
                )
            else:
                self.reports[sid].comm_time = 0
        else:
            self.reports[sid].comm_time = time.time() - self.reports[sid].comm_time

        # When the client is responding to an urgent request for an update, it will
        # store its (possibly different) client ID in its report
        client_id = self.reports[sid].client_id

        start_time = self.training_clients[client_id]["start_time"]
        finish_time = (
            self.reports[sid].training_time
            + self.reports[sid].processing_time
            + self.reports[sid].comm_time
            + start_time
        )
        starting_round = self.training_clients[client_id]["starting_round"]

        if Config().is_central_server():
            self.comm_overhead += self.reports[sid].edge_server_comm_overhead

        client_info = (
            finish_time,  # sorted by the client's finish time
            client_id,  # in case two or more clients have the same finish time
            {
                "client_id": client_id,
                "sid": sid,
                "starting_round": starting_round,
                "start_time": start_time,
                "report": self.reports[sid],
                "payload": self.client_payload[sid],
            },
        )

        if self.asynchronous_mode and self.simulate_wall_time:
            heapq.heappush(self.reported_clients, client_info)
        self.current_reported_clients[client_info[2]["client_id"]] = True
        del self.training_clients[client_id]

        self.training_sids.remove(client_info[2]["sid"])

        await self._process_clients(client_info)

    # pylint: disable=unused-argument
    def should_request_update(
        self, client_id, start_time, finish_time, client_staleness, report
    ):
        """Determines if an explicit request for model update should be sent to the client."""
        return client_staleness > self.staleness_bound and finish_time > self.wall_time

    async def _process_clients(self, client_info):
        """Determines whether it is time to process the client reports and
        proceed with the aggregation process.

        When in asynchronous mode, additional processing is needed to simulate
        the wall clock time.
        """
        # In asynchronous mode with simulated wall clock time, we need to extract
        # the minimum number of clients from the list of all reporting clients, and then
        # proceed with report processing and replace these clients with a new set of
        # selected clients
        if (
            self.asynchronous_mode
            and self.simulate_wall_time
            and len(self.current_reported_clients) >= len(self.selected_clients)
        ):
            # Step 1: Sanity checks to see if there are any stale clients; if so, send them
            # an urgent request for model updates at the current simulated wall clock time
            if self.request_update:
                # We should not proceed with further processing if there are outstanding requests
                # for urgent client updates
                for __, client_data in self.training_clients.items():
                    if client_data["update_requested"]:
                        return

                request_sent = False
                for i, client_info in enumerate(self.reported_clients):
                    client = client_info[2]
                    client_staleness = self.current_round - client["starting_round"]

                    if (
                        self.should_request_update(
                            client_id=client["client_id"],
                            start_time=client["start_time"],
                            finish_time=client_info[0],
                            client_staleness=client_staleness,
                            report=client["report"],
                        )
                        and not client["report"].update_response
                    ):
                        # Sending an urgent request to the client for a model update at the
                        # currently simulated wall clock time
                        client_id = client["client_id"]

                        logging.info(
                            "[Server #%s] Requesting urgent model update from client #%s.",
                            os.getpid(),
                            client_id,
                        )

                        # Remove the client information from the list of reporting clients since
                        # this client will report again soon with another model update upon
                        # receiving the request from the server
                        del self.reported_clients[i]

                        self.training_clients[client_id] = {
                            "id": client_id,
                            "starting_round": client["starting_round"],
                            "start_time": client["start_time"],
                            "update_requested": True,
                        }

                        sid = client["sid"]

                        self.training_sids.append(sid)

                        await self.sio.emit(
                            "request_update",
                            {
                                "client_id": client_id,
                                "time": self.wall_time - client["start_time"],
                            },
                            room=sid,
                        )
                        request_sent = True

                # If an urgent request was sent, we will wait until the client gets back to proceed
                # with aggregation.
                if request_sent:
                    return

            # Step 2: Processing clients in chronological order of finish times in wall clock time
            for __ in range(
                0, min(len(self.current_reported_clients), self.minimum_clients)
            ):
                # Extract a client with the earliest finish time in wall clock time
                client_info = heapq.heappop(self.reported_clients)
                client = client_info[2]

                # Removing from the list of current reporting clients as well, if needed
                self.current_processed_clients[client["client_id"]] = True

                # Update the simulated wall clock time to be the finish time of this client
                self.wall_time = client_info[0]

                # Add the report and payload of the extracted reporting client into updates
                logging.info(
                    "[Server #%s] Adding client #%s to the list of clients for aggregation.",
                    os.getpid(),
                    client["client_id"],
                )

                client_staleness = self.current_round - client["starting_round"]
                self.updates.append(
                    SimpleNamespace(
                        client_id=client["client_id"],
                        report=client["report"],
                        payload=client["payload"],
                        staleness=client_staleness,
                    )
                )

            # Step 3: Processing stale clients that exceed a staleness threshold

            # If there are more clients in the list of reporting clients that violate the
            # staleness bound, the server needs to wait for these clients even when the minimum
            # number of clients has been reached, by simply advancing its simulated wall clock
            # time ahead to include the remaining clients, until no stale clients exist
            possibly_stale_clients = []

            # Is there any reporting clients who are currently training on models that are too
            # `stale,` as defined by the staleness threshold? If so, we need to advance the wall
            # clock time until no stale clients exist in the future
            for __ in range(0, len(self.reported_clients)):
                # Extract a client with the earliest finish time in wall clock time
                client_info = heapq.heappop(self.reported_clients)
                heapq.heappush(possibly_stale_clients, client_info)

                if (
                    client_info[2]["starting_round"]
                    < self.current_round - self.staleness_bound
                ):
                    for __ in range(0, len(possibly_stale_clients)):
                        stale_client_info = heapq.heappop(possibly_stale_clients)
                        # Update the simulated wall clock time to be the finish time of this client
                        self.wall_time = stale_client_info[0]
                        client = stale_client_info[2]

                        # Add the report and payload of the extracted reporting client into updates
                        logging.info(
                            "[Server #%s] Adding client #%s to the list of clients for "
                            "aggregation.",
                            os.getpid(),
                            client["client_id"],
                        )

                        client_staleness = self.current_round - client["starting_round"]
                        self.updates.append(
                            SimpleNamespace(
                                client_id=client["client_id"],
                                report=client["report"],
                                payload=client["payload"],
                                staleness=client_staleness,
                            )
                        )

            self.reported_clients = possibly_stale_clients
            logging.info(
                "[Server #%s] Aggregating %s clients in total.",
                os.getpid(),
                len(self.updates),
            )

            await self._process_reports()
            await self.wrap_up()
            await self._select_clients()
            return

        if not self.simulate_wall_time or not self.asynchronous_mode:
            # In both synchronous and asynchronous modes, if we are not simulating the wall clock
            # time, we need to add the client report to the list of updates so far;
            # the same applies when we are running in synchronous mode.
            client = client_info[2]
            client_staleness = self.current_round - client["starting_round"]

            self.updates.append(
                SimpleNamespace(
                    client_id=client["client_id"],
                    report=client["report"],
                    payload=client["payload"],
                    staleness=client_staleness,
                )
            )

        if not self.simulate_wall_time:
            # In both synchronous and asynchronous modes, if we are not simulating the wall clock
            # time, it will need to be updated to the real wall clock time
            self.wall_time = time.time()

        if not self.asynchronous_mode and self.simulate_wall_time:
            # In synchronous mode with the wall clock time simulated, in addition to adding
            # the client report to the list of updates, we will also need to advance the wall
            # clock time to the finish time of the reporting client
            client_finish_time = client_info[0]
            self.wall_time = max(client_finish_time, self.wall_time)

            logging.info(
                "[%s] Advancing the wall clock time to %.2f.",
                self,
                self.wall_time,
            )

        # If all updates have been received from selected clients, the aggregation process
        # proceeds regardless of synchronous or asynchronous modes. This guarantees that
        # if asynchronous mode uses an excessively long aggregation interval, it will not
        # unnecessarily delay the aggregation process.
        if len(self.updates) >= self.clients_per_round:
            logging.info(
                "[%s] All %d client report(s) received. Processing.",
                self,
                len(self.updates),
            )
            await self._process_reports()
            await self.wrap_up()
            await self._select_clients()

        elif (
            hasattr(Config().trainer, "max_concurrency")
            and not Config().is_central_server()
        ):
            # Clients in the current batch finish training
            # The server will select the next batch of clients to train
            if len(self.updates) >= len(self.trained_clients) or len(
                self.current_reported_clients
            ) >= len(self.trained_clients):
                await self._select_clients(for_next_batch=True)

    async def _client_disconnected(self, sid):
        """When a client process disconnected it should be removed from its internal states."""
        for client_process_id, client in dict(self.clients).items():
            if client["sid"] == sid:
                # Obtain the client id before deleting
                client_id = self.clients[client_process_id]["client_id"]

                # Remove the physical client from server list
                del self.clients[client_process_id]
                logging.warning(
                    "[%s] Client process #%d disconnected and removed from this server, %d client processes are remaining.",
                    self,
                    client_process_id,
                    len(self.clients),
                )

                if len(self.clients) == 0:
                    logging.warning(
                        fonts.colourize(
                            f"[{self}] All clients disconnected, closing the server."
                        )
                    )
                    await self._close()

                # Handle the logical client under different situations
                if client_id in self.training_clients:
                    del self.training_clients[client_id]

                if client_id in self.current_reported_clients:
                    del self.current_reported_clients[client_id]

                # Decide continue or exit training
                if (
                    hasattr(Config(), "general")
                    and hasattr(Config().general, "debug")
                    and not Config().general.debug
                ):
                    # Recover from the failed client and proceed with training
                    if (
                        client_id in self.selected_clients
                        and client_id in self.trained_clients
                    ):
                        self.trained_clients.remove(client_id)
                        fail_client_index = self.selected_clients.index(client_id)
                        untrained_client_index = len(self.trained_clients)

                        # Swap current client to the begining of untrained clients
                        self.selected_clients[fail_client_index] = (
                            self.selected_clients[untrained_client_index]
                        )
                        self.selected_clients[untrained_client_index] = client_id

                        # Start next batch of client selection if current batch is done
                        if len(self.updates) >= len(self.trained_clients) or len(
                            self.current_reported_clients
                        ) >= len(self.trained_clients):
                            await self._select_clients(for_next_batch=True)
                else:
                    # Debug is either turned on or not specified, stop the training to avoid blocking.
                    logging.warning(
                        fonts.colourize(
                            f"[{self}] Closing the server due to a failed client."
                        )
                    )
                    await self._close()

    def save_to_checkpoint(self) -> None:
        """Saves a checkpoint for resuming the training session."""
        checkpoint_path = Config.params["checkpoint_path"]

        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        if "/" in model_name:
            model_name = model_name.replace("/", "_")
        filename = f"checkpoint_{model_name}_{self.current_round}.pth"
        logging.info(
            "[%s] Saving the checkpoint to %s/%s.",
            self,
            checkpoint_path,
            filename,
        )
        self.trainer.save_model(filename, checkpoint_path)
        self._save_random_states(self.current_round, checkpoint_path)

        # Saving the current round in the server for resuming its session later on
        with open(f"{checkpoint_path}/current_round.pkl", "wb") as checkpoint_file:
            pickle.dump(self.current_round, checkpoint_file)

    def _resume_from_checkpoint(self):
        """Resumes a training session from a previously saved checkpoint."""
        logging.info(
            "[%s] Resume a training session from a previously saved checkpoint.",
            self,
        )

        # Loading important data in the server for resuming its session
        checkpoint_path = Config.params["checkpoint_path"]

        with open(f"{checkpoint_path}/current_round.pkl", "rb") as checkpoint_file:
            self.current_round = pickle.load(checkpoint_file)

        self._restore_random_states(self.current_round, checkpoint_path)
        self.resumed_session = True

        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        filename = f"checkpoint_{model_name}_{self.current_round}.pth"
        self.trainer.load_model(filename, checkpoint_path)

    def _save_random_states(self, round_to_save, checkpoint_path):
        """Saves the random states in the server for resuming its session later on."""
        states_to_save = [
            f"numpy_prng_state_{round_to_save}",
            f"prng_state_{round_to_save}",
        ]

        variables_to_save = [
            np.random.get_state(),
            random.getstate(),
        ]

        for i, state in enumerate(states_to_save):
            with open(f"{checkpoint_path}/{state}.pkl", "wb") as checkpoint_file:
                pickle.dump(variables_to_save[i], checkpoint_file)

    def _restore_random_states(self, round_to_restore, checkpoint_path):
        """Restors the numpy.random and random states from previously saved checkpoints
        for a particular round.
        """
        states_to_load = ["numpy_prng_state", "prng_state"]
        variables_to_load = {}

        for i, state in enumerate(states_to_load):
            with open(
                f"{checkpoint_path}/{state}_{round_to_restore}.pkl", "rb"
            ) as checkpoint_file:
                variables_to_load[i] = pickle.load(checkpoint_file)

        numpy_prng_state = variables_to_load[0]
        self.prng_state = variables_to_load[1]

        np.random.set_state(numpy_prng_state)
        random.setstate(self.prng_state)

    async def wrap_up(self) -> None:
        """Wraps up when each round of training is done."""
        self.save_to_checkpoint()

        # Break the loop when the target accuracy is achieved
        target_accuracy = None
        target_perplexity = None

        if hasattr(Config().trainer, "target_accuracy"):
            target_accuracy = Config().trainer.target_accuracy
        elif hasattr(Config().trainer, "target_perplexity"):
            target_perplexity = Config().trainer.target_perplexity

        if target_accuracy and self.accuracy >= target_accuracy:
            logging.info("[%s] Target accuracy reached.", self)
            await self._close()

        if target_perplexity and self.accuracy <= target_perplexity:
            logging.info("[%s] Target perplexity reached.", self)
            await self._close()

        if self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")
            await self._close()

    async def _close(self):
        """Closes the server."""
        logging.info("[%s] Training concluded.", self)
        self.trainer.save_model()

        self.server_will_close()
        self.callback_handler.call_event("on_server_will_close", self)

        await self._close_connections()
        os._exit(0)

    def add_callbacks(self, callbacks):
        """Adds a list of callbacks to the server callback handler."""
        self.callback_handler.add_callbacks(callbacks)

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Customizes the server response with any additional information."""
        return server_response

    def customize_server_payload(self, payload):
        """Customizes the server payload before sending to the client."""
        return payload

    @abstractmethod
    async def _process_reports(self) -> None:
        """Processes a client report."""

    async def periodic_task(self) -> None:
        """
        Async method called periodically in asynchronous mode.
        """

    def clients_selected(self, selected_clients) -> None:
        """
        Method called after clients have been selected in each round."""

    def clients_processed(self) -> None:
        """Additional work to be performed after client reports have been processed."""

    def training_will_start(self) -> None:
        """
        Method called before selecting clients for the first round of training.
        """
        if Config().is_central_server():
            if self.disable_clients:
                logging.info("No clients are launched (server:disable_clients = true)")
            else:
                client_kwargs = None
                if getattr(Config().clients, "type", None) == "mpc":
                    client_kwargs = {
                        "round_store_lock": self._mpc_round_lock,
                        "debug_artifacts": getattr(
                            Config().clients, "mpc_debug_artifacts", False
                        ),
                    }
                Server._start_clients(client=self.client, client_kwargs=client_kwargs)

    def server_will_close(self) -> None:
        """
        Method called before closing the server.
        """
