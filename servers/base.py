"""
The base class for federated learning servers.
"""

from abc import abstractmethod, abstractstaticmethod
import os
import sys
import logging
import time
import pickle
import asyncio
import subprocess
from contextlib import closing
import multiprocessing as mp
import websockets

from config import Config
import client


class Server:
    """The base class for federated learning servers."""
    def __init__(self):
        self.clients = {}
        self.total_clients = 0
        self.selected_clients = None
        self.current_round = 0
        self.algorithm = None
        self.trainer = None
        self.accuracy = 0
        self.reports = []

    def run(self):
        """Start a run loop for the server. """
        # Remove the running trainers table from previous runs.
        with Config().sql_connection:
            with closing(Config().sql_connection.cursor()) as cursor:
                cursor.execute("DROP TABLE IF EXISTS trainers")

        self.configure()

        logging.info("Starting a server on port %s.", Config().server.port)
        start_server = websockets.serve(self.serve,
                                        Config().server.address,
                                        Config().server.port,
                                        ping_interval=None,
                                        max_size=2**30)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(start_server)

        if Config().is_central_server():
            # In cross-silo FL, the central server lets edge servers start first
            # Then starts their clients
            self.start_clients(as_server=True)

            # Allowing some time for the edge servers to start
            time.sleep(5)

        self.start_clients()
        loop.run_forever()

    def register_client(self, client_id, websocket):
        """Adding a newly arrived client to the list of clients."""
        if not client_id in self.clients:
            self.clients[client_id] = websocket

    def unregister_client(self, websocket):
        """Removing an existing client from the list of clients."""
        for key, value in dict(self.clients).items():
            if value == websocket:
                del self.clients[key]

    def start_clients(self, as_server=False):
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
                logging.info("Starting client #%s is an edge server.",
                             client_id)
                command = "python client.py -i {}".format(client_id)
                command += " -c {}".format(Config().args.config)
                command += " -p {}".format(Config().server.port + client_id)
                subprocess.Popen(command, shell=True)
            else:
                logging.info("Starting client #%s.", client_id)
                proc = mp.Process(target=client.run, args=(client_id, None))

                proc.start()

    async def close_connections(self):
        """Closing all WebSocket connections after training completes."""
        for _, client_socket in dict(self.clients).items():
            await client_socket.close()

    async def select_clients(self):
        """Select a subset of the clients and send messages to them to start training."""
        self.reports = []
        self.current_round += 1

        logging.info("\n[Server #%d] Starting round %s/%s.", os.getpid(),
                     self.current_round,
                     Config().trainer.rounds)

        self.choose_clients()

        if len(self.selected_clients) > 0:
            for client_id in self.selected_clients:
                socket = self.clients[client_id]
                logging.info("[Server #%d] Selecting client #%s for training.",
                             os.getpid(), client_id)
                server_response = {'id': client_id, 'payload': True}

                server_response = await self.customize_server_response(
                    server_response)
                # Sending the server response as metadata to the clients (payload to follow)
                await socket.send(pickle.dumps(server_response))

                payload = self.algorithm.extract_weights()
                payload = await self.customize_server_payload(payload)

                # Sending the server payload to the clients
                await self.send(socket, payload)

    async def send(self, socket, payload):
        """Sending the client payload to the server using WebSockets."""
        logging.info("[Server #%d] Sending the current model.", os.getpid())
        if isinstance(payload, list):
            data_size = 0

            for data in payload:
                _data = pickle.dumps(data)
                await socket.send(_data)
                data_size += sys.getsizeof(_data)
        else:
            _data = pickle.dumps(payload)
            await socket.send(_data)
            data_size = sys.getsizeof(_data)

        logging.info("[Server #%d] Sent %s bytes of payload data.",
                     os.getpid(), data_size)

    async def serve(self, websocket, path):  # pylint: disable=unused-argument
        """Running a federated learning server."""
        try:
            async for message in websocket:
                data = pickle.loads(message)
                client_id = data['id']
                logging.info("[Server #%d] Data received from client #%s.",
                             os.getpid(), client_id)

                if 'payload' in data:
                    # an existing client reports new updates from local training
                    report = data['report']
                    payload = await self.recv(client_id, report, websocket)
                    self.reports.append((report, payload))

                    if len(self.reports) == len(self.selected_clients):
                        logging.info(
                            "[Server #%d] All client reports received. Processing.",
                            os.getpid())
                        await self.process_reports()
                        await self.wrap_up()
                        await self.select_clients()
                else:
                    # a new client arrives
                    self.register_client(client_id, websocket)

                    if self.current_round == 0 and len(
                            self.clients) >= self.total_clients:
                        logging.info("[Server #%d] Starting training.",
                                     os.getpid())
                        await self.select_clients()
        except websockets.ConnectionClosed as exception:
            logging.info(
                "[Server #%d] WebSockets connection closed abnormally.",
                os.getpid())
            logging.error(exception)
            sys.exit()

    async def recv(self, client_id, client_report, websocket):
        """Receiving the payload from a client using WebSockets."""
        logging.info("[Server #%d] Receiving payload data from client #%s.",
                     os.getpid(), client_id)

        if hasattr(client_report, 'payload_length'):
            client_payload = []
            payload_size = 0
            for __ in range(0, client_report.payload_length):
                _data = await websocket.recv()
                payload = pickle.loads(_data)
                client_payload.append(payload)
                payload_size += sys.getsizeof(_data)
        else:
            _data = await websocket.recv()
            client_payload = pickle.loads(_data)
            payload_size = sys.getsizeof(_data)

        logging.info(
            "[Server #%d] Received %s bytes of payload data from client #%s.",
            os.getpid(), payload_size, client_id)

        return client_payload

    async def wrap_up(self):
        """Wrapping up when each round of training is done."""
        # Break the loop when the target accuracy is achieved
        target_accuracy = Config().trainer.target_accuracy

        if target_accuracy and self.accuracy >= target_accuracy:
            logging.info("Target accuracy reached.")
            await self.close()

        if self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")
            await self.close()

    async def close(self):
        """Closing the server."""
        self.trainer.save_model()
        await self.close_connections()
        self.trainer.stop_training()
        sys.stdout.flush()
        sys.exit()

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
    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""

    @abstractstaticmethod
    def is_valid_server_type(server_type):
        """Determine if the server type is valid. """
