"""
The base class for federated learning servers.
"""

from abc import abstractmethod
import json
import logging
import subprocess
import pickle


class Server():
    """The base class for federated learning servers."""

    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.selected_clients = None
        self.current_round = 0
        self.model = None
        self.reports = []
        self.dataset_type = config.training.dataset
        self.data_path = '{}/{}'.format(config.training.data_path, config.training.dataset)


    def register_client(self, client_id, websocket):
        """Adding a newly arrived client to the list of clients."""
        if not client_id in self.clients:
            self.clients[client_id] = websocket


    def unregister_client(self, websocket):
        """Removing an existing client from the list of clients."""
        for key, value in dict(self.clients).items():
            if value == websocket:
                del self.clients[key]


    def start_clients(self):
        """Starting all the clients as separate processes."""
        for client_id in range(1, self.config.clients.total + 1):
            logging.info("Starting client #%s...", client_id)
            command = "python client.py -i {}".format(client_id)
            subprocess.Popen(command, shell=True)


    async def select_clients(self):
        """Select a subset of the clients and send messages to them to start training."""
        self.reports = []
        self.current_round += 1
        logging.info('**** Round %s/%s ****', self.current_round, self.config.training.rounds)

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
        """Perform consecutive rounds of federated learning training."""

        total_rounds = self.config.training.rounds
        target_accuracy = self.config.training.target_accuracy
        logging.info("Starting training on %s clients...", self.config.clients.per_round)

        if target_accuracy:
            logging.info('Training: %s rounds or %s%% accuracy\n',
                total_rounds, 100 * target_accuracy)
        else:
            logging.info('Training: %s rounds\n', total_rounds)

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

                    # Break loop when target accuracy is met
                    if target_accuracy and (accuracy >= target_accuracy):
                        logging.info('Target accuracy reached.')
                        return

                    await self.select_clients()

            else:
                # a new client arrives
                self.register_client(client_id, websocket)

                if self.current_round == 0 and len(self.clients) >= self.config.clients.total:
                    logging.info('Starting FL training...')
                    await self.select_clients()


    @abstractmethod
    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""


    @abstractmethod
    def process_report(self):
        """Process the reports after all clients have sent them to the server."""
