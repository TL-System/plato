"""
The base class for federated learning servers.
"""

from abc import abstractmethod
import json
import sys
import os
import logging
import subprocess
import pickle
import websockets

from config import Config


class Server:
    """The base class for federated learning servers."""
    def __init__(self):
        self.clients = {}
        self.total_clients = 0
        self.selected_clients = None
        self.current_round = 0
        self.model = None
        self.trainer = None
        self.accuracy = 0
        self.reports = []

    def register_client(self, client_id, websocket):
        """Adding a newly arrived client to the list of clients."""
        if not client_id in self.clients:
            self.clients[client_id] = websocket

    def unregister_client(self, websocket):
        """Removing an existing client from the list of clients."""
        for key, value in dict(self.clients).items():
            if value == websocket:
                del self.clients[key]

    @staticmethod
    def start_clients(as_server=False):
        """Starting all the clients as separate processes."""
        starting_id = 1

        if as_server:
            total_processes = Config().cross_silo.total_silos
            starting_id += Config().clients.total_clients
        else:
            total_processes = Config().clients.total_clients

        for client_id in range(starting_id, total_processes + starting_id):
            logging.info("Starting client #%s...", client_id)
            command = "python client.py -i {}".format(client_id)
            command += " -c {}".format(Config.args.config)

            if as_server:
                command += " -p {}".format(Config().server.port + client_id)
                logging.info("This client #%s is an edge server.", client_id)

            subprocess.Popen(command, shell=True)

    async def close_connections(self):
        """Closing all WebSocket connections after training completes."""
        for _, client_socket in dict(self.clients).items():
            await client_socket.close()

    async def select_clients(self):
        """Select a subset of the clients and send messages to them to start training."""
        self.reports = []
        self.current_round += 1

        logging.info('\n[Server %d] Starting round %s/%s.', os.getpid(),
                     self.current_round,
                     Config().trainer.rounds)

        self.choose_clients()

        if len(self.selected_clients) > 0:
            for client_id in self.selected_clients:
                socket = self.clients[client_id]
                logging.info(
                    "[Server %d] Selecting client #%s for training...",
                    os.getpid(), client_id)
                server_response = {'id': client_id, 'payload': True}
                server_response = await self.wrap_up_server_response(
                    server_response)
                await socket.send(json.dumps(server_response))

                logging.info("Sending the current model...")
                await socket.send(pickle.dumps(self.trainer.extract_weights()))

    async def serve(self, websocket, path):  # pylint: disable=unused-argument
        """Running a federated learning server."""
        try:
            async for message in websocket:
                data = json.loads(message)
                client_id = data['id']
                logging.info("[Server %s] Data received from client #%s",
                             os.getpid(), client_id)

                if 'payload' in data:
                    # an existing client reports new updates from local training
                    client_update = await websocket.recv()
                    report = pickle.loads(client_update)
                    logging.info(
                        "[Server %s] Update from client #%s received.",
                        os.getpid(), client_id)

                    self.wrap_up_client_report(report)
                    self.reports.append(report)

                    if len(self.reports) == len(self.selected_clients):
                        await self.process_reports()
                        await self.wrap_up()
                        await self.select_clients()
                else:
                    # a new client arrives
                    self.register_client(client_id, websocket)

                    if self.current_round == 0 and len(
                            self.clients) >= self.total_clients:
                        logging.info('[Server %s] Starting FL training.',
                                     os.getpid())
                        await self.select_clients()
        except websockets.ConnectionClosed as exception:
            logging.info("Server %s: WebSockets connection closed abnormally.",
                         os.getpid())
            logging.error(exception)
            sys.exit()

    async def wrap_up(self):
        """Wrapping up when each round of training is done."""
        if not Config().is_edge_server():
            # Break the loop when the target accuracy is achieved
            target_accuracy = Config().trainer.target_accuracy

            if target_accuracy and self.accuracy >= target_accuracy:
                logging.info('Target accuracy reached.')
                self.trainer.save_model()
                await self.close_connections()
                sys.exit()
            if self.current_round >= Config().trainer.rounds:
                logging.info('Target number of training rounds reached.')
                self.trainer.save_model()
                await self.close_connections()
                sys.exit()

    async def wrap_up_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        return server_response

    def wrap_up_client_report(self, report):
        """Wrap up after receiving the client report with any additional information."""

    @abstractmethod
    def configure(self):
        """Configuring the server with initialization work."""

    @abstractmethod
    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""

    @abstractmethod
    async def process_reports(self):
        """Process the reports after all clients have sent them to the server."""
