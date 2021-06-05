"""
The base class for federated learning servers.
"""

import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
from abc import abstractmethod

import socketio
from aiohttp import web
from plato.client import run
from plato.config import Config


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

    async def on_client_payload(self, sid, data):
        """ An existing client sends a new payload from local training. """
        await self.plato_server.client_payload_arrived(sid, data['payload'])

    async def on_client_payload_done(self, sid, data):
        """ An existing client finished sending its payloads from local training. """
        await self.plato_server.client_payload_done(sid, data['id'])


class Server:
    """The base class for federated learning servers."""
    def __init__(self):
        self.sio = None
        self.client = None
        self.clients = {}
        self.total_clients = 0
        self.clients_per_round = 0
        self.selected_clients = None
        self.current_round = 0
        self.algorithm = None
        self.trainer = None
        self.accuracy = 0
        self.reports = {}
        self.updates = []
        self.client_payload = {}

    def run(self, client=None):
        """Start a run loop for the server. """
        # Remove the running trainers table from previous runs.
        if not Config().is_edge_server():
            with Config().sql_connection:
                Config().cursor.execute("DROP TABLE IF EXISTS trainers")

        self.client = client
        self.configure()

        if Config().is_central_server():
            # In cross-silo FL, the central server lets edge servers start first
            # Then starts their clients
            Server.start_clients(as_server=True)

            # Allowing some time for the edge servers to start
            time.sleep(5)

        Server.start_clients(client=self.client)

        self.start()

    def start(self, port=Config().server.port):
        """ Start running the socket.io server. """
        logging.info("Starting a server at address %s and port %s.",
                     Config().server.address, port)

        ping_interval = Config().server.ping_interval if hasattr(
            Config().server, 'ping_interval') else 3600
        self.sio = socketio.AsyncServer(ping_interval=ping_interval,
                                        max_http_buffer_size=2**31)
        self.sio.register_namespace(
            ServerEvents(namespace='/', plato_server=self))
        app = web.Application()
        self.sio.attach(app)
        web.run_app(app, host=Config().server.address, port=port)

    async def register_client(self, sid, client_id):
        """Adding a newly arrived client to the list of clients."""
        if not client_id in self.clients:
            # The last contact time is stored for each client
            self.clients[client_id] = {
                'sid': sid,
                'last_contacted': time.time()
            }
            logging.info("[Server #%d] New client with id #%d arrived.",
                         os.getpid(), client_id)
        else:
            self.clients[client_id]['last_contacted'] = time.time()
            logging.info("[Server #%d] New contact from Client #%d received.",
                         os.getpid(), client_id)

        if self.current_round == 0 and len(
                self.clients) >= self.clients_per_round:
            logging.info("[Server #%d] Starting training.", os.getpid())
            await self.select_clients()

    @staticmethod
    def start_clients(client=None, as_server=False):
        """Starting all the clients as separate processes."""
        starting_id = 1

        if as_server:
            total_processes = Config().algorithm.total_silos
            starting_id += Config().clients.total_clients
        else:
            total_processes = Config().clients.total_clients

        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)

        for client_id in range(starting_id, total_processes + starting_id):
            if as_server:
                port = int(Config().server.port) + client_id
                logging.info(
                    "Starting client #%d as an edge server on port %s.",
                    client_id, port)
                proc = mp.Process(target=run, args=(client_id, port, client))
                proc.start()
            else:
                logging.info("Starting client #%d's process.", client_id)
                proc = mp.Process(target=run, args=(client_id, None, client))
                proc.start()

    async def close_connections(self):
        """Closing all socket.io connections after training completes."""
        for client_id, client in dict(self.clients).items():
            logging.info("Closing the connection to client #%d.", client_id)
            await self.sio.emit('disconnect', room=client['sid'])

    async def select_clients(self):
        """Select a subset of the clients and send messages to them to start training."""
        self.updates = []
        self.current_round += 1

        logging.info("\n[Server #%d] Starting round %s/%s.", os.getpid(),
                     self.current_round,
                     Config().trainer.rounds)

        self.selected_clients = self.choose_clients()

        if len(self.selected_clients) > 0:
            for client_id in self.selected_clients:
                sid = self.clients[client_id]['sid']
                await self.register_client(sid, client_id)

                logging.info("[Server #%d] Selecting client #%d for training.",
                             os.getpid(), client_id)

                server_response = {'id': client_id}
                server_response = await self.customize_server_response(
                    server_response)

                # Sending the server response as metadata to the clients (payload to follow)
                await self.sio.emit('payload_to_arrive',
                                    {'response': server_response},
                                    room=sid)

                payload = self.algorithm.extract_weights()
                payload = await self.customize_server_payload(payload)

                # Sending the server payload to the client
                await self.send(sid, payload, client_id)

    async def send(self, sid, payload, client_id):
        """ Sending the client payload to the server using socket.io. """
        logging.info("[Server #%d] Sending the current model to client #%d.",
                     os.getpid(), client_id)
        if isinstance(payload, list):
            data_size = 0

            for data in payload:
                _data = pickle.dumps(data)
                await self.sio.emit('payload', {'data': _data}, room=sid)
                data_size += sys.getsizeof(_data)
        else:
            _data = pickle.dumps(payload)
            await self.sio.emit('payload', {'data': _data}, room=sid)
            data_size = sys.getsizeof(_data)

        await self.sio.emit('payload_done', {'id': client_id}, room=sid)

        logging.info("[Server #%d] Sent %s MB of payload data to client #%d.",
                     os.getpid(), round(data_size / 1024**2, 2), client_id)

    async def client_report_arrived(self, sid, report):
        """ Upon receiving a report from a client. """
        self.reports[sid] = pickle.loads(report)
        self.client_payload[sid] = None

    async def client_payload_arrived(self, sid, payload):
        """ Upon receiving a portion of the payload from a client. """
        _data = pickle.loads(payload)

        if self.client_payload[sid] is None:
            self.client_payload[sid] = _data
        elif isinstance(self.client_payload[sid], list):
            self.client_payload[sid].append(_data)
        else:
            self.client_payload[sid] = [self.client_payload[sid]]
            self.client_payload[sid].append(_data)

    async def client_payload_done(self, sid, client_id):
        """ Upon receiving all the payload from a client. """
        assert self.client_payload[sid] is not None

        payload_size = 0
        if isinstance(self.client_payload[sid], list):
            for _data in self.client_payload[sid]:
                payload_size += sys.getsizeof(pickle.dumps(_data))
        else:
            payload_size = sys.getsizeof(pickle.dumps(
                self.client_payload[sid]))

        logging.info(
            "[Server #%d] Received %s MB of payload data from client #%d.",
            os.getpid(), round(payload_size / 1024**2, 2), client_id)

        self.updates.append((self.reports[sid], self.client_payload[sid]))

        if len(self.updates) > 0 and len(self.updates) >= len(
                self.selected_clients):
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
        """Wrapping up when each round of training is done."""
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
        """Closing the server."""
        logging.info("[Server #%d] Training concluded.", os.getpid())
        self.trainer.save_model()
        await self.close_connections()
        os._exit(0)

    async def customize_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        return server_response

    async def customize_server_payload(self, payload):
        """Wrap up generating the server payload with any additional information."""
        return payload

    @abstractmethod
    def configure(self):
        """Configuring the server with initialization work."""

    @abstractmethod
    def choose_clients(self) -> list:
        """Choose a subset of the clients to participate in each round."""
