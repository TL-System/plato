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
        if 's3_key' in data:
            await self.plato_server.client_payload_done(sid,
                                                        data['id'],
                                                        s3_key=data['s3_key'])
        else:
            await self.plato_server.client_payload_done(sid, data['id'])


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

        # Clients whose new reports were received but not yet processed
        self.reporting_clients = []

        # Clients who are still training since the last round of aggregation
        self.training_clients = {}

        # The wall clock time that is simulated to accommodate the fact that
        # clients can only run a batch at a time, controlled by `max_concurrency`
        self.wall_time = time.time()

        # When simulating the wall clock time, the server needs to remember the
        # set of reporting clients received since the previous round of aggregation
        self.current_reporting_clients = []

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
            periodic_interval = Config().server.periodic_interval
        else:
            periodic_interval = 5

        asyncio.get_event_loop().create_task(self.periodic(periodic_interval))

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
        web.run_app(app,
                    host=Config().server.address,
                    port=port,
                    loop=asyncio.get_event_loop())

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

        # When simulating the wall clock time, if len(self.reporting_clients) is 0, the
        # server has aggregated all reporting clients already
        if hasattr(Config().server, 'synchronous') and not Config(
        ).server.synchronous and self.selected_clients is not None and len(
                self.reporting_clients) > 0 and len(
                    self.reporting_clients) < self.clients_per_round:
            # If self.selected_clients is None, it implies that it is the first iteration;
            # If len(self.reporting_clients) == self.clients_per_round, it implies that
            # all selected clients have already reported.

            # Except for these two cases, we need to exclude the clients who are still
            # training.
            training_client_ids = [
                self.training_clients[client_id]['id']
                for client_id in list(self.training_clients.keys())
            ]

            # If the server is simulating the wall clock time, some of the clients who
            # reported may not have been aggregated; they should be excluded from the next
            # round of client selection
            reporting_client_ids = [
                client[1] for client in self.reporting_clients
            ]

            selectable_clients = [
                client for client in self.clients_pool
                if client not in training_client_ids
                and client not in reporting_client_ids
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

                self.training_clients[client_id] = {
                    'id': selected_client_id,
                    'starting_round': self.current_round,
                    'start_time': self.wall_time
                }

            # There is no need to clear the list of reporting clients if we are
            # simulating the wall clock time on the server. This is because
            # when wall clock time is simulated, the server needs to wait for
            # all the clients to report before selecting a subset of clients for
            # replacement, and all remaining reporting clients will be processed
            # in the next round
            if hasattr(Config().server, "simulate_wall_time") and Config(
            ).server.simulate_wall_time:
                self.current_reporting_clients = []
                return

            self.reporting_clients = []

    def choose_clients(self, clients_pool, clients_count):
        """ Choose a subset of the clients to participate in each round. """
        assert clients_count <= len(clients_pool)

        # Select clients randomly
        return random.sample(clients_pool, clients_count)

    async def periodic(self, periodic_interval):
        """ Runs periodic_task() periodically on the server. The time interval between
            its execution is defined in 'server:periodic_interval'.
        """
        while True:
            await self.periodic_task()
            await asyncio.sleep(periodic_interval)

    async def periodic_task(self):
        """ A periodic task that is executed from time to time, determined by
        'server:periodic_interval' with a default value of 5 seconds, in the configuration. """
        # Call the async function that defines a customized periodic task, if any
        _task = getattr(self, "customize_periodic_task", None)
        if callable(_task):
            await self.customize_periodic_task()

        simulate_wall_time = hasattr(
            Config().server,
            'simulate_wall_time') and Config().server.simulate_wall_time

        # If we are operating in asynchronous mode, aggregate the model updates received so far.
        if not simulate_wall_time and hasattr(
                Config().server,
                'synchronous') and not Config().server.synchronous:

            # What is the minimum number of clients that must have reported before aggregation
            # takes place?
            minimum_clients = 1
            if hasattr(Config().server, 'minimum_clients_aggregated'):
                minimum_clients = Config().server.minimum_clients_aggregated

            # Is there any training clients who are currently training on models that are too
            # `stale,` as defined by the staleness threshold?
            staleness = 0
            if hasattr(Config().server, 'staleness'):
                staleness = Config().server.staleness

            for __, client_data in self.training_clients.items():
                # The client is still working at an early round, early enough to stop the aggregation
                # process as determined by 'staleness'
                if client_data[
                        'starting_round'] < self.current_round - staleness:
                    logging.info(
                        "[Server #%d] Client %s is still working at round %s, which is "
                        "beyond the staleness threshold %s compared to the current round %s. "
                        "Nothing to process.", os.getpid(), client_data['id'],
                        client_data['starting_round'], staleness,
                        self.current_round)

                    return

            if len(self.updates) >= minimum_clients:
                logging.info(
                    "[Server #%d] %d client reports received in asynchronous mode. Processing.",
                    os.getpid(), len(self.updates))
                await self.process_reports()
                await self.wrap_up()
                await self.select_clients()
            else:
                logging.info(
                    "[Server #%d] Simulating wall clock time or there are no sufficient number "
                    "of client reports have been received. Nothing to process.",
                    os.getpid())

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

        metadata = {'id': client_id}

        if self.s3_client is not None:
            s3_key = f'server_payload_{os.getpid()}_{self.current_round}'
            self.s3_client.send_to_s3(s3_key, payload)
            data_size = sys.getsizeof(pickle.dumps(payload))
            metadata['s3_key'] = s3_key
        else:
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

        await self.sio.emit('payload_done', metadata, room=sid)

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

    async def client_payload_done(self, sid, client_id, s3_key=None):
        """ Upon receiving all the payload from a client, either via S3 or socket.io. """
        if s3_key is None:
            assert self.client_payload[sid] is not None

            payload_size = 0
            if isinstance(self.client_payload[sid], list):
                for _data in self.client_payload[sid]:
                    payload_size += sys.getsizeof(pickle.dumps(_data))
            else:
                payload_size = sys.getsizeof(
                    pickle.dumps(self.client_payload[sid]))
        else:
            self.client_payload[sid] = self.s3_client.receive_from_s3(s3_key)
            payload_size = sys.getsizeof(pickle.dumps(
                self.client_payload[sid]))

        logging.info(
            "[Server #%d] Received %s MB of payload data from client #%d.",
            os.getpid(), round(payload_size / 1024**2, 2), client_id)

        # Pass through the inbound_processor(s), if any
        self.client_payload[sid] = self.inbound_processor.process(
            self.client_payload[sid])

        start_time = self.training_clients[client_id]['start_time']
        finish_time = self.reports[sid].training_time + start_time
        starting_round = self.training_clients[client_id]['starting_round']

        client_info = (
            finish_time,  # sorted by the client's finish time
            {
                'client_id': client_id,
                'starting_round': starting_round,
                'start_time': start_time,
                'report': self.reports[sid],
                'payload': self.client_payload[sid],
            })

        heapq.heappush(self.reporting_clients, client_info)
        self.current_reporting_clients.append(client_info)
        del self.training_clients[client_id]

        await self.process_clients()

    async def process_clients(self):
        """ Determine whether it is time to process the client reports and
            proceed with the aggregation process.

            When in asynchronous mode, additional processing is needed to simulate
            the wall clock time.
        """
        asynchronous_mode = hasattr(
            Config().server, "synchronous") and not Config().server.synchronous
        simulate_wall_time = hasattr(
            Config().server,
            "simulate_wall_time") and Config().server.simulate_wall_time
        minimum_clients = 1
        if hasattr(Config().server, 'minimum_clients_aggregated'):
            minimum_clients = Config().server.minimum_clients_aggregated

        # In asynchronous mode with simulated wall clock time, we need to extract
        # the minimum number of clients from the list of all reporting clients, and then
        # proceed with report processing and replace these clients with a new set of
        # selected clients
        if asynchronous_mode and simulate_wall_time and len(
                self.current_reporting_clients) >= len(self.selected_clients):
            # Step 1: Sanity checks to see if there are any stale clients; if so, send them
            # an urgent request for model updates at the current simulated wall clock time
            staleness = 0
            if hasattr(Config().server, 'staleness'):
                staleness = Config().server.staleness

            if hasattr(Config().server,
                       'request_update') and Config().server.request_update:
                request_sent = False
                for i, client_info in enumerate(self.reporting_clients):
                    if client_info[1][
                            'starting_round'] < self.current_round - staleness and not client_info[
                                1]['report'].update_response:

                        # Sending an urgent request to the client for a model update at the
                        # currently simulated wall clock time
                        client_id = client_info[1]['client_id']

                        logging.info(
                            "[Server #%s] Requesting urgent model update from client %s.",
                            os.getpid(), client_id)

                        # Remove the client information from the list of reporting clients since
                        # this client will report again soon with another model update upon
                        # receiving the request from the server
                        del self.reporting_clients[i]

                        sid = self.clients[client_id]['sid']

                        self.training_clients[client_id] = {
                            'id': client_id,
                            'starting_round': client_info[1]['starting_round'],
                            'start_time': client_info[1]['start_time']
                        }

                        await self.sio.emit('request_update',
                                            {'time': self.wall_time},
                                            room=sid)
                        request_sent = True

                if request_sent:
                    return

            # Step 2: Processing clients in chronological order of finish times in wall clock time
            for __ in range(
                    0, min(len(self.current_reporting_clients),
                           minimum_clients)):
                # Extract a client with the earliest finish time in wall clock time
                client_info = heapq.heappop(self.reporting_clients)
                # Update the simulated wall clock time to be the finish time of this client
                self.wall_time = client_info[0]

                # Add the report and payload of the extracted reporting client into updates
                logging.info(
                    "[Server #%s] Adding client #%s to the list of clients for aggregation.",
                    os.getpid(), client_info[1]['client_id'])

                self.updates.append(
                    (client_info[1]['report'], client_info[1]['payload']))

            # Step 3: Processing stale clients that exceed a staleness threshold

            # If there are more clients in the list of reporting clients that violate the
            # staleness bound, the server needs to wait for these clients even when the minimum
            # number of clients has been reached, by simply advancing its simulated wall clock
            # time ahead to include the remaining clients, until no stale clients exist
            possibly_stale_clients = []

            # Is there any reporting clients who are currently training on models that are too
            # `stale,` as defined by the staleness threshold? If so, we need to advance the wall
            # clock time until no stale clients exist in the future
            for __ in range(0, len(self.reporting_clients)):
                # Extract a client with the earliest finish time in wall clock time
                client_info = heapq.heappop(self.reporting_clients)
                heapq.heappush(possibly_stale_clients, client_info)

                if client_info[1][
                        'starting_round'] < self.current_round - staleness:
                    for __ in range(0, len(possibly_stale_clients)):
                        stale_client_info = heapq.heappop(
                            possibly_stale_clients)
                        # Update the simulated wall clock time to be the finish time of this client
                        self.wall_time = stale_client_info[0]

                        # Add the report and payload of the extracted reporting client into updates
                        logging.info(
                            "[Server #%s] Adding client #%s to the list of clients for "
                            "aggregation.", os.getpid(),
                            stale_client_info[1]['client_id'])
                        self.updates.append((stale_client_info[1]['report'],
                                             stale_client_info[1]['payload']))

            self.reporting_clients = possibly_stale_clients
            logging.info("[Server #%s] Aggregating %s clients in total.",
                         os.getpid(), len(self.updates))

            await self.process_reports()
            await self.wrap_up()
            await self.select_clients()

        # If all updates have been received from selected clients, the aggregation process
        # proceeds regardless of synchronous or asynchronous modes. This guarantees that
        # if asynchronous mode uses an excessively long aggregation interval, it will not
        # unnecessarily delay the aggregation process.
        elif len(self.reporting_clients) >= self.clients_per_round:
            logging.info(
                "[Server #%d] All %d client reports received. Processing.",
                os.getpid(), len(self.reporting_clients))

            # Add the report and payload of all reporting clients into updates
            for client_info in self.reporting_clients:
                self.updates.append(
                    (client_info[1]['report'], client_info[1]['payload']))

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

                    if len(self.updates) >= len(self.selected_clients):
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
