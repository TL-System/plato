"""
The base class for federated learning servers.
"""

from abc import abstractmethod
import json
import logging
import subprocess
import sys
import pickle


class Server():
    """The base class for federated learning servers."""

    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.model = None
        self.reports = []
        self.dataset_type = config.training.dataset
        self.data_path = '{}/{}'.format(config.training.data_path, config.training.dataset)


    def register_client(self, client_id, websocket):
        if not client_id in self.clients:
            self.clients[client_id] = websocket

        logging.info("clients: %s", self.clients)


    def unregister_client(self, websocket):
        for key, value in dict(self.clients).items():
            if value == websocket:
                del self.clients[key]

        logging.info("clients: %s", self.clients)


    def start_clients(self):
        """Starting all the clients as separate processes."""
        for client_id in range(1, self.config.clients.total + 1):
            logging.info("Starting client #%s...", client_id)
            command = "python client.py -i {}".format(client_id)
            subprocess.Popen(command, shell=True)


    async def wait_for_clients(self, websocket):
        """Waiting for clients to arrive."""
        async for message in websocket:
            data = json.loads(message)
            client_id = data['id']
            logging.info("New client arrived with ID: %s", client_id)

            # a new client arrives
            assert 'payload' not in data
            self.register_client(client_id, websocket)

            if len(self.clients) == self.config.clients.total:
                return


    async def one_round(self, websocket):
        """
        Selecting some clients to participate in the current round,
        and run them for one round.
        """
        selected_clients = self.select_clients()
        self.reports = []

        assert len(selected_clients) > 0

        # If the list of selected_clients is not empty
        for client_id in selected_clients:
            socket = self.clients[client_id]
            logging.info("Selecting client with ID %s...", client_id)
            server_response = {'id': client_id, 'payload': True}
            await socket.send(json.dumps(server_response))

            logging.info("Sending the current model...")
            await socket.send(pickle.dumps(self.model.state_dict()))

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

                if len(self.reports) == len(selected_clients):
                    return self.process_report()
            else:
                # a new client arrives
                self.register_client(client_id, websocket)


    async def serve(self, websocket, path):
        """Perform consecutive rounds of federated learning training."""
        rounds = self.config.training.rounds
        target_accuracy = self.config.training.target_accuracy

        await self.wait_for_clients(websocket)

        logging.info("Starting training on %s clients...", self.config.clients.per_round)

        if target_accuracy:
            logging.info('Training: %s rounds or %s%% accuracy\n',
                rounds, 100 * target_accuracy)
        else:
            logging.info('Training: %s rounds\n', rounds)

        for current_round in range(1, rounds + 1):
            logging.info('**** Round %s/%s ****', current_round, rounds)

            # Run one federated learning round
            accuracy = await self.one_round(websocket)

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break


    @abstractmethod
    def select_clients(self):
        """Select devices to participate in round."""
        pass


    @abstractmethod
    def aggregate_weights(self, reports):
        """Aggregate the reported weight updates from the selected clients."""
        pass


    @abstractmethod
    def process_report(self):
        """Process the reports after all clients have sent them to the server."""
        pass