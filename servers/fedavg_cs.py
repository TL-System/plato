"""
A cross-silo federated learning server using federated averaging, as either edge or central servers.
"""

import logging
import time
import os
import asyncio

from servers import FedAvgServer
from config import Config
from utils import csv_processor


class FedAvgCrossSiloServer(FedAvgServer):
    """Cross-silo federated learning server using federated averaging."""
    def __init__(self):
        super().__init__()

        self.current_global_round = 1

        if Config().is_edge_server():
            # An edge client waits for the event that a certain number of
            # aggregations are completed
            self.model_aggregated = asyncio.Event()

            # An edge client waits for the event that a new global round begins
            # before starting the first round of local aggregation
            self.new_global_round_begins = asyncio.Event()

            # Compute the number of clients in each silo for edge servers
            self.total_clients = int(self.total_clients /
                                     Config().cross_silo.total_silos)
            self.clients_per_round = int(self.clients_per_round /
                                         Config().cross_silo.total_silos)
            logging.info(
                "Edge server #%s starts training with %s clients and %s per round...",
                Config().args.id, self.total_clients, self.clients_per_round)

            if Config().results:
                self.recorded_items = ['global_round'] + self.recorded_items
                result_dir = f'./results/{Config().training.dataset}/{Config().training.model}/{Config().server.type}/'
                result_csv_file = f'{result_dir}result_{Config().args.id}.csv'
                csv_processor.initialize_csv(result_csv_file,
                                             self.recorded_items, result_dir)

        # Compute the number of clients for the central server
        if Config().is_central_server():
            self.clients_per_round = Config().cross_silo.total_silos
            self.total_clients = self.clients_per_round

            logging.info(
                "The central server starts training with %s edge servers...",
                self.total_clients)

    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        if Config().args.id:
            logging.info('Configuring edge server #%s as a %s server...',
                         Config().args.id,
                         Config().server.type)
            logging.info('Training with %s local aggregation rounds.',
                         Config().cross_silo.rounds)

        else:
            logging.info('Configuring the %s server...', Config().server.type)

            total_rounds = Config().training.rounds
            target_accuracy = Config().training.target_accuracy

            if target_accuracy:
                logging.info('Training: %s rounds or %s%% accuracy\n',
                             total_rounds, 100 * target_accuracy)
            else:
                logging.info('Training: %s rounds\n', total_rounds)

        self.load_test_data()
        self.load_model()

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        if Config().results:
            # Write results into a CSV file
            result_dir = f'./results/{Config().training.dataset}/{Config().training.model}/{Config().server.type}/'

            new_row = []
            for item in self.recorded_items:
                item_value = {
                    'global_round': self.current_global_round,
                    'round': self.current_round,
                    'accuracy': self.accuracy * 100,
                    'training_time': time.time() - self.round_start_time,
                    'edge_agg_num': Config().cross_silo.rounds
                }[item]
                new_row.append(item_value)

            if Config().is_edge_server():
                result_csv_file = f'{result_dir}result_{Config().args.id}.csv'
            else:
                result_csv_file = f'{result_dir}result.csv'

            csv_processor.write_csv(result_csv_file, new_row)

        # When a certain number of aggregations are completed, an edge client
        # may need to be signaled to send a report to the central server
        if Config().is_edge_server():
            if self.current_round == Config().cross_silo.rounds:
                logging.info(
                    '[Server %s] Completed %s rounds of local aggregation.',
                    os.getpid(),
                    Config().cross_silo.rounds)
                self.model_aggregated.set()

                self.current_round = 0
                self.new_global_round_begins.clear()
                # Wait until a new global round begins
                # to avoid selecting clients before a new global round begins
                await self.new_global_round_begins.wait()
                self.current_global_round += 1
