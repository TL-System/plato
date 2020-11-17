"""
A simple edge server supporting hierarchical federated learning.
Edge servers are also clients to FedAvgServer as the root servers.
Each edge server communicates with a subset of the clients within its own institution.
"""

import logging
import random
import json
import pickle
from dataclasses import dataclass
import websockets

from training import trainer
from servers import FedAvgServer
from servers.fedcs import CrossSiloServer
from config import Config


@dataclass
class Report:
    """Edge server report sent to the federated learning central server."""
    edge_id: str
    total_samples: int
    weights: list
    accuracy: float


class EdgeServer(FedAvgServer):
    """Federated learning edge server using federated averaging."""

    def __init__(self):
        super().__init__()
        self.edge_id = Config().args.edgeid
        self.clients = {} # Clients of this edge server
        self.do_test = None # Should edge servers test the trained model?
        self.testset = None # Testing dataset
        self.report = None # Report to the central server
        self.model = None # Machine learning model
        self.aggregations = Config().cross_silo.aggregations # Aggregation number on edge servers in one global training round
        self.current_agg_round = 0

        self.clients_id = CrossSiloServer().assign_clients_to_edge_servers()[self.edge_id]
        self.clients_num = len(self.clients_id)
        self.clients_per_round = CrossSiloServer().select_clients_for_edge_servers()[self.edge_id]
        self.client_reports = [] # Reports from clients


    def configure(self):
        """
        Booting the federated learning edge server by setting up the data, model, and
        creating the clients.
        """
        logging.info('Configuring the edge server #%s...', self.edge_id)

        self.assign_clients()

        self.load_test_data()
        self.load_model()


    def assign_clients(self):
        """Assign clients connected to this edge server."""




    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""
        # Select clients randomly
        assert self.clients_per_round <= self.clients_num
        self.selected_clients = random.sample(list(self.clients), self.clients_per_round)


    async def select_clients(self):
        """Select a subset of its clients and send messages to them to start training."""
        self.client_reports = []
        self.current_agg_round += 1
        logging.info('**** Aggreagtion round %s/%s ****',
                    self.current_agg_round, Config().cross_silo.aggregations)

        self.choose_clients()

        if len(self.selected_clients) > 0:
            for client_id in self.selected_clients:
                socket = self.clients[client_id]
                logging.info("Selecting client with ID %s for training...", client_id)
                server_response = {'id': client_id, 'payload': True}
                await socket.send(json.dumps(server_response))

                logging.info("Sending the current model...")
                await socket.send(pickle.dumps(self.model.state_dict()))


    async def run(self, websocket, path):
        """Running a federated learning edge server for one round of global training."""

        logging.info("Waiting for %s clients to arrive...", self.clients_num)
        logging.info("Path: %s", path)

        try:
            async for message in websocket:
                data = json.loads(message)
                client_id = data['id']
                logging.info("client data received with ID: %s", client_id)

                if 'payload' in data:
                    # an existing client reports new updates from local training
                    client_update = await websocket.recv()
                    client_report = pickle.loads(client_update)
                    logging.info("Client update received. Accuracy = {:.2f}%\n"
                        .format(100 * client_report.accuracy))

                    self.client_reports.append(client_report)

                    if len(self.client_reports) == len(self.selected_clients):
                        total_samples, weights, accuracy = self.process_report()

                        if self.current_agg_round == self.aggregations:
                            # generate the report that will be sent to the central server
                            self.report = Report(self.edge_id, total_samples, weights, accuracy)
                            self.current_agg_round = 0

                        await self.select_clients()
                else:
                    # a new client arrives
                    self.register_client(client_id, websocket)

                    if self.current_agg_round == 0 and len(self.clients) >= self.clients_num:
                        logging.info('Starting FL training on edge server #%s...', self.edge_id)
                        await self.select_clients()
        except websockets.ConnectionClosed as exception:
            logging.info("Edge Server WebSockets connection closed abnormally.")
            logging.error(exception)


    async def start_edge_server(self, websocket_with_clients, edge_server_path):
        """Startup function for an edge server."""
        uri = 'ws://{}:{}'.format(Config().server.address, Config().server.port)
        try:
            async with websockets.connect(uri, ping_interval=None, max_size=2 ** 30) as websocket:
                logging.info("Signing in at the central server with edge server ID %s...",
                            self.edge_id)
                await websocket.send(json.dumps({'id': self.edge_id}))

                while True:
                    logging.info("Edge server %s is waiting for a new round of global training...",
                                self.edge_id)
                    server_response = await websocket.recv()
                    data = json.loads(server_response)

                    if data['id'] == self.edge_id and 'payload' in data:
                        logging.info("Edge server %s is receiving the current global model...",
                                    self.edge_id)
                        server_model = await websocket.recv()
                        self.model.load_state_dict(pickle.loads(server_model))

                        await self.run(websocket_with_clients, edge_server_path)

                        logging.info("Model aggregated on edge server #%s.", self.edge_id)
                        # Sending edge server ID as metadata to the server (payload to follow)
                        edge_update = {'id': self.edge_id, 'payload': True}
                        await websocket.send(json.dumps(edge_update))

                        # Sending the edge server report to the central server as payload
                        await websocket.send(pickle.dumps(self.report))


        except OSError as exception:
            logging.info("Edge Server #%s: connection to the central server failed.", self.edge_id)
            logging.error(exception)


    def process_reports_from_clients(self):
        """Process the client reports by aggregating their weights."""
        updated_weights = self.aggregate_weights(self.client_reports)
        total_samples = sum([client_report.num_samples for client_report in self.client_reports])
        trainer.load_weights(self.model, updated_weights)

        # Testing the aggregated model accuracy
        if Config().clients.do_test:
            # Compute the average accuracy from client reports
            accuracy = self.accuracy_averaging(self.client_reports)
            logging.info('Average client accuracy: {:.2f}%\n'.format(100 * accuracy))
        elif Config().cross_silo.do_test:
            # Test the updated model directly at the edge server
            accuracy = trainer.test(self.model, self.testset, Config().training.batch_size)
            logging.info('Aggregated model accuracy: {:.2f}%\n'.format(100 * accuracy))

        return total_samples, updated_weights, accuracy
