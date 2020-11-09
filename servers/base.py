"""
The base class for federated learning servers.
"""

from abc import abstractmethod
import json
import logging
import pickle

class Server():
    """The base class for federated learning servers."""

    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.model = None
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


    async def serve(self, websocket, path):
        try:
            async for message in websocket:
                data = json.loads(message)
                client_id = data['id']
                self.register_client(client_id, websocket)
                logging.info("client data received with ID: %s", client_id)

                if 'payload' in data:
                    client_update = await websocket.recv()
                    report = pickle.loads(client_update)
                    logging.info("Client update received. Accuracy = %s", report.accuracy)

                    # Aggregate client reports
                    reports = []
                    reports.append(report)
                    self.aggregate_weights(reports)
            
                # Select a client with a particular client ID
                logging.info("Selecting client with ID %s...", client_id)
                server_response = {'id': client_id, 'payload': True}
                await websocket.send(json.dumps(server_response))

                logging.info("Sending the current model after a round of aggregation...")
                # Send the current model after a round of aggregation, as payload
                await websocket.send(pickle.dumps(self.model.state_dict()))
        finally:
            logging.info("Closing client connection...")
            # self.unregister_client(websocket)


    def run(self):
        """Perform consecutive rounds of federated learning training."""
        rounds = self.config.training.rounds
        target_accuracy = self.config.training.target_accuracy

        if target_accuracy:
            logging.info('Training: %s rounds or %s%% accuracy\n',
                rounds, 100 * target_accuracy)
        else:
            logging.info('Training: %s rounds\n', rounds)

        # Perform rounds of federated learning
        for current_round in range(1, rounds + 1):
            logging.info('**** Round %s/%s ****', current_round, rounds)

            # Run the federated learning round
            accuracy = self.round()

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

    @abstractmethod
    def round(self):
        """
        Selecting some clients to participate in the current round,
        and run them for one round.
        """
        pass

    @abstractmethod
    def aggregate_weights(self, reports):
        """
        Selecting some clients to participate in the current round,
        and run them for one round.
        """
        pass
