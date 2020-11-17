"""
A simple edge server supporting hierarchical federated learning.
Edge servers are also clients to FedAvgServer as the root servers.
Each edge server communicates with a subset of the clients within its own institution.
"""

import logging
import json
import pickle
import subprocess
import sys
import websockets

from servers import FedAvgServer
from config import Config


class CrossSiloServer(FedAvgServer):
    """Cross silo federated learning central server."""

    def __init__(self):
        super().__init__()
        self.silos = {}


    def register_edge_servers(self, edge_id, websocket):
        """Adding a newly arrived client to the list of edge_servers."""
        if not edge_id in self.silos:
            self.silos[edge_id] = websocket


    def unregister_edge_servers(self, websocket):
        """Removing an existing client from the list of edge_servers."""
        for key, value in dict(self.silos).items():
            if value == websocket:
                del self.silos[key]


    async def close_connections(self):
        """Closing all WebSocket connections after training completes."""
        for _, edge_socket in dict(self.silos).items():
            await edge_socket.close()
        for _, client_socket in dict(self.clients).items():
            await client_socket.close()


    def assign_clients_to_edge_servers(self):
        """Assign clients to edge servers."""
        clients_id_list = {}

        edges_num = Config().cross_silo.total_silos
        clients_num = Config().clients.total_clients
        per_clients_num = int(clients_num / edges_num)
        residual_clients_num = int(clients_num % edges_num)

        assigned_clients_num = 0

        # Assign (almost) the same number of clients to edge servers
        for edge_id in range(1, edges_num + 1):
            if edge_id <= residual_clients_num:
                clients_id_list[edge_id] = [assigned_clients_num + i for i in range(1, per_clients_num + 2)]
            else:
                clients_id_list[edge_id] = [assigned_clients_num + i for i in range(1, per_clients_num + 1)]

            assigned_clients_num += len(clients_id_list[edge_id])

        return clients_id_list


    def select_clients_for_edge_servers(self):
        """Determine the number of clients attending each round for each edge server."""
        clients_per_round_list = {}

        edges_num = Config().cross_silo.total_silos
        clients_per_round_num = Config().clients.per_round
        per_clients_num = int(clients_per_round_num / edges_num)
        residual_clients_num = int(clients_per_round_num % edges_num)

        assigned_clients_num = 0

        for edge_id in range(1, edges_num + 1):
            if edge_id <= residual_clients_num:
                clients_per_round_list[edge_id] = [assigned_clients_num + per_clients_num + 1]
            else:
                clients_per_round_list[edge_id] = [assigned_clients_num + per_clients_num]

            assigned_clients_num += clients_per_round_list[edge_id]

        return clients_per_round_list


    def configure(self):
        """
        Booting the cross-silo federated learning central server by setting up the data, model, and
        creating the edge servers and clients.
        """

        logging.info('Configuring the %s server...', Config().server.type)

        total_rounds = Config().training.rounds
        target_accuracy = Config().training.target_accuracy

        if target_accuracy:
            logging.info('Training: %s rounds or %s%% accuracy\n',
                total_rounds, 100 * target_accuracy)
        else:
            logging.info('Training: %s rounds\n', total_rounds)

        logging.info("Starting training on %s edge servers and %s clients in total...",
            Config().cross_silo.total_silos, Config().clients.per_round)

        self.load_test_data()
        self.load_model()


    async def serve(self, websocket, path):
        """Running a cross-silo federated learning server."""

        logging.info("Waiting for %s edge servers to arrive...", Config().cross_silo.total_silos)
        logging.info("Waiting for %s clients to arrive...", Config().clients.total_clients)
        logging.info("Path: %s", path)

        self.configure()

        try:
            async for message in websocket:
                data = json.loads(message)
                edge_id = data['id']
                logging.info("Edge server data received with ID: %s", edge_id)

                if 'payload' in data:
                    # An existing edge server reports new updates from local aggregation
                    edge_update = await websocket.recv()
                    report = pickle.loads(edge_update)
                    logging.info("Edge server update received. Accuracy = {:.2f}%\n"
                        .format(100 * report.accuracy))

                    self.reports.append(report)

                    if len(self.reports) == len(self.silos):
                        accuracy = self.process_report()

                        # Break the loop when the target accuracy is achieved
                        target_accuracy = Config().training.target_accuracy

                        if target_accuracy and (accuracy >= target_accuracy):
                            logging.info('Target accuracy reached.')
                            await self.close_connections()
                            sys.exit()

                        if self.current_round == Config().training.rounds:
                            logging.info('Target number of training rounds reached.')
                            await self.close_connections()
                            sys.exit()

                        await self.select_clients()

                else:
                    # a new edge server arrives
                    self.register_edge_servers(edge_id, websocket)

                    if self.current_round == 0 and len(self.silos) >= Config().cross_silo.total_silos:
                        logging.info('Starting cross-silo FL training...')

        except websockets.ConnectionClosed as exception:
            logging.info("Server WebSockets connection closed abnormally.")
            logging.error(exception)
