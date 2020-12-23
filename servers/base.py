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
import asyncio
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
        self.accuracy = 0
        self.reports = []

        # Directory of results (figures etc.)
        self.result_dir = './results/' + Config(
        ).training.dataset + '/' + Config().training.model + '/'

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

        logging.info('[Server %d] Starting round %s/%s.', os.getpid(),
                     self.current_round,
                     Config().training.rounds)

        self.choose_clients()

        if len(self.selected_clients) > 0:
            for client_id in self.selected_clients:
                socket = self.clients[client_id]
                logging.info("Selecting client #%s for training...", client_id)
                server_response = {'id': client_id, 'payload': True}
                if Config().rl and not Config().args.id:
                    server_response = await self.generate_rl_info(
                        server_response)
                await socket.send(json.dumps(server_response))

                logging.info("Sending the current model...")
                await socket.send(pickle.dumps(self.model.state_dict()))

    async def serve(self, websocket, path):
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
                        "Server {}: Update from client #{} received. Accuracy = {:.2f}%."
                        .format(os.getpid(), client_id, 100 * report.accuracy))

                    self.reports.append(report)

                    if len(self.reports) == len(self.selected_clients):
                        self.accuracy = self.process_reports()

                        await self.wrap_up_one_round()

                        # Break the loop when the target accuracy is achieved
                        target_accuracy = Config().training.target_accuracy

                        if not Config().args.port:
                            if target_accuracy and self.accuracy >= target_accuracy:
                                logging.info('Target accuracy reached.')
                                self.wrap_up()
                                await self.close_connections()
                                sys.exit()

                            if self.current_round >= Config().training.rounds:
                                logging.info(
                                    'Target number of training rounds reached.'
                                )
                                self.wrap_up()
                                await self.close_connections()
                                sys.exit()

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

    def wrap_up(self):
        """Wrapping up when the training is done."""

    @abstractmethod
    def configure(self):
        """Configuring the server with initialization work."""

    @abstractmethod
    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""

    @abstractmethod
    def process_reports(self):
        """Process the reports after all clients have sent them to the server."""

    @abstractmethod
    async def wrap_up_one_round(self):
        """Wrapping up when one round of training is done."""
