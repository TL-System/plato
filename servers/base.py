"""
The base class for federated learning servers.
"""

from abc import abstractmethod
import json
import sys
import logging
import subprocess
import pickle
import websockets

from config import Config


class Server():
    """The base class for federated learning servers."""

    def __init__(self):
        self.clients = {}
        self.selected_clients = None
        self.current_round = 0
        self.model = None
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


    def start_clients(self, as_server=False):
        """Starting all the clients as separate processes."""
        if as_server:
            total_processes = Config().cross_silo.total_silos
            arg = '-e'
        else:
            total_processes = Config().clients.total_clients
            arg = '-i'

        for client_id in range(1, total_processes + 1):
            logging.info("Starting client #%s...", client_id)
            command = "python client.py {} {}".format(arg, client_id)
            subprocess.Popen(command, shell=True)


    async def close_connections(self):
        """Closing all WebSocket connections after training completes."""
        for _, client_socket in dict(self.clients).items():
            await client_socket.close()


    async def select_clients(self):
        """Select a subset of the clients and send messages to them to start training."""
        self.reports = []
        self.current_round += 1
        logging.info('**** Round %s/%s ****', self.current_round, Config().training.rounds)

        self.choose_clients()

        if len(self.selected_clients) > 0:
            for client_id in self.selected_clients:
                socket = self.clients[client_id]
                logging.info("Selecting client with ID %s for training...", client_id)
                server_response = {'id': client_id, 'payload': True}
                await socket.send(json.dumps(server_response))

                logging.info("Sending the current model...")
                await socket.send(pickle.dumps(self.model.state_dict()))


    async def serve(self, websocket, path):
        """Running a federated learning server."""

        logging.info("Waiting for %s clients to arrive...", Config().clients.total_clients)
        logging.info("Path: %s", path)

        self.configure()

        try:
            async for message in websocket:
                data = json.loads(message)
                client_id = data['id']
                logging.info("client data received with ID: %s", client_id)

                if 'payload' in data:
                    # an existing client reports new updates from local training
                    client_update = await websocket.recv()
                    report = pickle.loads(client_update)
                    logging.info("Client update received. Accuracy = {:.2f}%\n"
                        .format(100 * report.accuracy))

                    self.reports.append(report)

                    if len(self.reports) == len(self.selected_clients):
                        accuracy = self.process_report()

                        # Break the loop when the target accuracy is achieved
                        target_accuracy = Config().training.target_accuracy

                        if target_accuracy and (accuracy >= target_accuracy):
                            logging.info('Target accuracy reached.')
                            await self.close_connections()
                            sys.exit()

                        if self.current_round >= Config().training.rounds:
                            logging.info('Target number of training rounds reached.')
                            await self.close_connections()
                            sys.exit()

                        await self.select_clients()

                else:
                    # a new client arrives
                    self.register_client(client_id, websocket)

                    if self.current_round == 0 and len(self.clients) >= Config().clients.total_clients:
                        logging.info('Starting FL training...')
                        await self.select_clients()
        except websockets.ConnectionClosed as exception:
            logging.info("Server WebSockets connection closed abnormally.")
            logging.error(exception)


    @abstractmethod
    def configure(self):
        """Configuring the server with initialization work."""


    @abstractmethod
    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""


    @abstractmethod
    def process_report(self):
        """Process the reports after all clients have sent them to the server."""
