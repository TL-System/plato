"""
The base class for federated learning servers.
"""

import asyncio
import logging
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from abc import abstractmethod

import socketio
from aiohttp import web

from plato.client import run
from plato.config import Config
from plato.utils import s3


class ServerEvents(socketio.AsyncNamespace):
    """ A custom namespace for socketio.AsyncServer. """
    def __init__(self, namespace, plato_server):
        super().__init__(namespace)
        self.plato_server = plato_server

    #pylint: disable=unused-argument
    async def on_connect(self, sid, environ):
        """ Upon a new connection from a client. """
        logging.info("[Server #%d] A new client just connected.", os.getpid())

    async def on_disconnect(self, sid):
        """ Upon a disconnection event. """
        logging.info("[Server #%d] An existing client just disconnected.",
                     os.getpid())
        await self.plato_server.client_disconnected(sid)

    async def on_client_alive(self, sid, data):
        """ A new client arrived or an existing client sends a heartbeat. """
        await self.plato_server.register_client(sid, data['id'])

    async def on_client_report(self, sid, data):
        """ An existing client sends a new report from local training. """
        await self.plato_server.client_report_arrived(sid, data['report'])

    async def on_chunk(self, sid, data):
        """ A chunk of data from the server arrived. """
        await self.plato_server.client_chunk_arrived(sid, data['data'])

    async def on_client_payload(self, sid, data):
        """ An existing client sends a new payload from local training. """
        await self.plato_server.client_payload_arrived(sid, data['id'])

    async def on_client_payload_done(self, sid, data):
        """ An existing client finished sending its payloads from local training. """
        await self.plato_server.client_payload_done(sid, data['id'],
                                                    data['obkey'])


class Server:
    """ The base class for federated learning servers. """
    def __init__(self):
        self.sio = None
        self.client = None
        self.clients = {}
        self.total_clients = 0
        # The client ids are stored for client selection
        self.clients_pool = []
        self.clients_per_round = 0
        self.selected_clients = None
        self.current_round = 0
        self.algorithm = None
        self.trainer = None
        self.accuracy = 0
        self.reports = {}
        self.updates = []
        self.client_payload = {}
        self.client_chunks = {}
        self.s3_client = None
        self.outbound_processor = None
        self.inbound_processor = None

        # States that need to be maintained for asynchronous FL

        # Clients whose new reports were received since the last round of aggregation
        self.reporting_clients = []
        # Clients who are still training since the last round of aggregation
        self.training_clients = {}

    def run(self,
            client=None,
            edge_server=None,
            edge_client=None,
            trainer=None):
        """ Start a run loop for the server. """
        # Remove the running trainers table from previous runs.
        if not Config().is_edge_server() and hasattr(Config().trainer,
                                                     'max_concurrency'):
            with Config().sql_connection:
                Config().cursor.execute("DROP TABLE IF EXISTS trainers")

        self.client = client
        self.configure()

        if Config().is_central_server():
            # In cross-silo FL, the central server lets edge servers start first
            # Then starts their clients
            Server.start_clients(as_server=True,
                                 client=self.client,
                                 edge_server=edge_server,
                                 edge_client=edge_client,
                                 trainer=trainer)

            # Allowing some time for the edge servers to start
            time.sleep(5)

        if hasattr(Config().server,
                   'disable_clients') and Config().server.disable_clients:
            logging.info(
                "No clients are launched (server:disable_clients = true)")
        else:
            Server.start_clients(client=self.client)

        if hasattr(Config().server, 'periodic_interval'):
            asyncio.get_event_loop().create_task(self.periodic())

        self.start()

    def start(self, port=Config().server.port):
        """ Start running the socket.io server. """
        logging.info("Starting a server at address %s and port %s.",
                     Config().server.address, port)

        ping_interval = Config().server.ping_interval if hasattr(
            Config().server, 'ping_interval') else 3600
        ping_timeout = Config().server.ping_timeout if hasattr(
            Config().server, 'ping_timeout') else 360
        self.sio = socketio.AsyncServer(ping_interval=ping_interval,
                                        max_http_buffer_size=2**31,
                                        ping_timeout=ping_timeout)
        self.sio.register_namespace(
            ServerEvents(namespace='/', plato_server=self))

        if hasattr(Config().server, 's3_endpoint_url'):
            self.s3_client = s3.S3()

        app = web.Application()
        self.sio.attach(app)
        web.run_app(app, host=Config().server.address, port=port)

    async def register_client(self, sid, client_id):
        """ Adding a newly arrived client to the list of clients. """
        if not client_id in self.clients:
            # The last contact time is stored for each client
            self.clients[client_id] = {
                'sid': sid,
                'last_contacted': time.perf_counter()
            }
            logging.info("[Server #%d] New client with id #%d arrived.",
                         os.getpid(), client_id)
        else:
            self.clients[client_id]['last_contacted'] = time.perf_counter()
            logging.info("[Server #%d] New contact from Client #%d received.",
                         os.getpid(), client_id)

        if self.current_round == 0 and len(
                self.clients) >= self.clients_per_round:
            logging.info("[Server #%d] Starting training.", os.getpid())
            await self.select_clients()

    @staticmethod
    def start_clients(client=None,
                      as_server=False,
                      edge_server=None,
                      edge_client=None,
                      trainer=None):
        """ Starting all the clients as separate processes. """
        starting_id = 1

        if hasattr(Config().clients,
                   'simulation') and Config().clients.simulation:
            # In the client simulation mode, we only need to launch a limited
            # number of client objects (same as the number of clients per round)
            client_processes = Config().clients.per_round
        else:
            client_processes = Config().clients.total_clients

        if as_server:
            total_processes = Config().algorithm.total_silos
            starting_id += client_processes
        else:
            total_processes = client_processes

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)

        for client_id in range(starting_id, total_processes + starting_id):
            if as_server:
                port = int(Config().server.port) + client_id
                logging.info(
                    "Starting client #%d as an edge server on port %s.",
                    client_id, port)
                proc = mp.Process(target=run,
                                  args=(client_id, port, client, edge_server,
                                        edge_client, trainer))
                proc.start()
            else:
                logging.info("Starting client #%d's process.", client_id)
                proc = mp.Process(target=run,
                                  args=(client_id, None, client, None, None,
                                        None))
                proc.start()

    async def close_connections(self):
        """ Closing all socket.io connections after training completes. """
        for client_id, client in dict(self.clients).items():
            logging.info("Closing the connection to client #%d.", client_id)
            await self.sio.emit('disconnect', room=client['sid'])

    async def select_clients(self):
        """ Select a subset of the clients and send messages to them to start training. """
        self.updates = []
        self.current_round += 1

        logging.info("\n[Server #%d] Starting round %s/%s.", os.getpid(),
                     self.current_round,
                     Config().trainer.rounds)

        if hasattr(Config().clients, 'simulation') and Config(
        ).clients.simulation and not Config().is_central_server:
            # In the client simulation mode, the client pool for client selection contains
            # all the virtual clients to be simulated
            self.clients_pool = list(range(1, 1 + self.total_clients))

        else:
            # If no clients are simulated, the client pool for client selection consists of
            # the current set of clients that have contacted the server
            self.clients_pool = list(self.clients)

        # In asychronous FL, avoid selecting new clients to replace those that are still
        # training at this time
        if hasattr(Config().server, 'synchronous') and not Config(
        ).server.synchronous and self.selected_clients is not None and len(
                self.reporting_clients) < self.clients_per_round:
            # If self.selected_clients is None, it implies that it is the first iteration;
            # If len(self.reporting_clients) == self.clients_per_round, it implies that
            # all selected clients have already reported.

            # Except for these two cases, we need to exclude the clients who are still
            # training.
            training_client_ids = [
                self.training_clients[client_id]
                for client_id in list(self.training_clients.keys())
            ]
            selectable_clients = [
                client for client in self.clients_pool
                if client not in training_client_ids
            ]

            self.selected_clients = self.choose_clients(
                selectable_clients, len(self.reporting_clients))
        else:
            self.selected_clients = self.choose_clients(
                self.clients_pool, self.clients_per_round)

        if len(self.selected_clients) > 0:
            for i, selected_client_id in enumerate(self.selected_clients):
                if hasattr(Config().clients, 'simulation') and Config(
                ).clients.simulation and not Config().is_central_server:
                    if hasattr(Config().server, 'synchronous') and not Config(
                    ).server.synchronous and self.reporting_clients is not None:
                        client_id = self.reporting_clients[i]
                    else:
                        client_id = i + 1
                else:
                    client_id = selected_client_id

                sid = self.clients[client_id]['sid']

                logging.info("[Server #%d] Selecting client #%d for training.",
                             os.getpid(), selected_client_id)

                server_response = {'id': selected_client_id}
                server_response = await self.customize_server_response(
                    server_response)

                # Sending the server response as metadata to the clients (payload to follow)
                await self.sio.emit('payload_to_arrive',
                                    {'response': server_response},
                                    room=sid)

                payload = self.algorithm.extract_weights()
                payload = self.customize_server_payload(payload)

                # Sending the server payload to the client
                logging.info(
                    "[Server #%d] Sending the current model to client #%d.",
                    os.getpid(), selected_client_id)
                await self.send(sid, payload, selected_client_id)

                self.training_clients[client_id] = selected_client_id

            self.reporting_clients = []

    def choose_clients(self, clients_pool, clients_count):
        """ Choose a subset of the clients to participate in each round. """
        assert clients_count <= len(clients_pool)

        # Select clients randomly
        return random.sample(clients_pool, clients_count)

    async def periodic(self):
        """ Runs periodic_task() periodically on the server. The time interval between
            its execution is defined in 'server:periodic_interval'.
        """
        while True:
            await self.periodic_task()
            await asyncio.sleep(Config().server.periodic_interval)

    async def periodic_task(self):
        """ A periodic task that is executed from time to time, determined by
        'server:periodic_interval' in the configuration. """
        # Call the async function that defines a customized periodic task, if any
        _task = getattr(self, "customize_periodic_task", None)
        if callable(_task):
            await self.customize_periodic_task()

        # If we are operating in asynchronous mode, aggregate the model updates received so far.
        if hasattr(Config().server,
                   'synchronous') and not Config().server.synchronous:
            if len(self.updates) > 0:
                logging.info(
                    "[Server #%d] %d client reports received in asynchronous mode. Processing.",
                    os.getpid(), len(self.updates))
                await self.process_reports()
                await self.wrap_up()
                await self.select_clients()
            else:
                logging.info(
                    "[Server #%d] No client reports have been received. Nothing to process."
                )

    async def send_in_chunks(self, data, sid, client_id) -> None:
        """ Sending a bytes object in fixed-sized chunks to the client. """
        step = 1024 ^ 2
        chunks = [data[i:i + step] for i in range(0, len(data), step)]

        for chunk in chunks:
            await self.sio.emit('chunk', {'data': chunk}, room=sid)

        await self.sio.emit('payload', {'id': client_id}, room=sid)

    async def send(self, sid, payload, client_id) -> None:
        """ Sending a new data payload to the client using either S3 or socket.io. """
        # First apply outbound processors, if any
        payload = self.outbound_processor.process(payload)

        if self.s3_client is not None:
            payload_key = f'server_payload_{os.getpid()}_{self.current_round}'
            self.s3_client.send_to_s3(payload_key, payload)
            data_size = sys.getsizeof(pickle.dumps(payload))
        else:
            payload_key = None
            data_size = 0

            if isinstance(payload, list):
                for data in payload:
                    _data = pickle.dumps(data)
                    await self.send_in_chunks(_data, sid, client_id)
                    data_size += sys.getsizeof(_data)

            else:
                _data = pickle.dumps(payload)
                await self.send_in_chunks(_data, sid, client_id)
                data_size = sys.getsizeof(_data)

        await self.sio.emit('payload_done', {
            'id': client_id,
            'obkey': payload_key
        },
                            room=sid)

        logging.info("[Server #%d] Sent %s MB of payload data to client #%d.",
                     os.getpid(), round(data_size / 1024**2, 2), client_id)

    async def client_report_arrived(self, sid, report):
        """ Upon receiving a report from a client. """
        self.reports[sid] = pickle.loads(report)
        self.client_payload[sid] = None
        self.client_chunks[sid] = []

    async def client_chunk_arrived(self, sid, data) -> None:
        """ Upon receiving a chunk of data from a client. """
        self.client_chunks[sid].append(data)

    async def client_payload_arrived(self, sid, client_id):
        """ Upon receiving a portion of the payload from a client. """
        assert len(
            self.client_chunks[sid]) > 0 and client_id in self.training_clients

        payload = b''.join(self.client_chunks[sid])
        _data = pickle.loads(payload)
        self.client_chunks[sid] = []

        if self.client_payload[sid] is None:
            self.client_payload[sid] = _data
        elif isinstance(self.client_payload[sid], list):
            self.client_payload[sid].append(_data)
        else:
            self.client_payload[sid] = [self.client_payload[sid]]
            self.client_payload[sid].append(_data)

    async def client_payload_done(self, sid, client_id, object_key):
        """ Upon receiving all the payload from a client, either via S3 or socket.io. """
        if object_key is None:
            assert self.client_payload[sid] is not None

            payload_size = 0
            if isinstance(self.client_payload[sid], list):
                for _data in self.client_payload[sid]:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            else:
                payload_size = sys.getsizeof(
                    pickle.dumps(self.client_payload[sid]))
        else:
            self.client_payload[sid] = self.s3_client.receive_from_s3(
                object_key)
            payload_size = sys.getsizeof(pickle.dumps(
                self.client_payload[sid]))

        logging.info(
            "[Server #%d] Received %s MB of payload data from client #%d.",
            os.getpid(), round(payload_size / 1024**2, 2), client_id)

        self.client_payload[sid] = self.inbound_processor.process(
            self.client_payload[sid])
        self.updates.append((self.reports[sid], self.client_payload[sid]))

        self.reporting_clients.append(client_id)
        del self.training_clients[client_id]

        # If all updates have been received from selected clients, the aggregation process
        # proceeds regardless of synchronous or asynchronous modes. This guarantees that
        # if asynchronous mode uses an excessively long aggregation interval, it will not
        # unnecessarily delay the aggregation process.
        if len(self.updates) > 0 and len(
                self.updates) >= self.clients_per_round:
            logging.info(
                "[Server #%d] All %d client reports received. Processing.",
                os.getpid(), len(self.updates))
            await self.process_reports()
            await self.wrap_up()
            await self.select_clients()

    async def client_disconnected(self, sid):
        """ When a client disconnected it should be removed from its internal states. """
        for client_id, client in dict(self.clients).items():
            if client['sid'] == sid:
                del self.clients[client_id]

                if client_id in self.training_clients:
                    del self.training_clients[client_id]

                logging.info(
                    "[Server #%d] Client #%d disconnected and removed from this server.",
                    os.getpid(), client_id)

                if client_id in self.selected_clients:
                    self.selected_clients.remove(client_id)

                    if len(self.updates) > 0 and len(self.updates) >= len(
                            self.selected_clients):
                        logging.info(
                            "[Server #%d] All %d client reports received. Processing.",
                            os.getpid(), len(self.updates))
                        await self.process_reports()
                        await self.wrap_up()
                        await self.select_clients()

    async def wrap_up(self):
        """ Wrapping up when each round of training is done. """
        # Break the loop when the target accuracy is achieved
        target_accuracy = Config().trainer.target_accuracy

        if target_accuracy and self.accuracy >= target_accuracy:
            logging.info("[Server #%d] Target accuracy reached.", os.getpid())
            await self.close()

        if self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")
            await self.close()

    # pylint: disable=protected-access
    async def close(self):
        """ Closing the server. """
        logging.info("[Server #%d] Training concluded.", os.getpid())
        self.trainer.save_model()
        await self.close_connections()
        os._exit(0)

    @abstractmethod
    def configure(self):
        """ Configuring the server with initialization work. """

    async def customize_server_response(self, server_response):
        """ Wrap up generating the server response with any additional information. """
        return server_response

    @abstractmethod
    def customize_server_payload(self, payload):
        """ Wrap up generating the server payload with any additional information. """

    @abstractmethod
    async def process_reports(self) -> None:
        """ Process a client report. """
