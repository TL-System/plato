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

        self.current_global_round = None

        if hasattr(Config(), 'results'):
            self.recorded_items = ['global_round'] + self.recorded_items

            dataset = Config().data.dataset
            model = Config().trainer.model
            server_type = Config().algorithm.type
            result_dir = f'./results/{dataset}/{model}/{server_type}/'
            csv_file = f'{result_dir}result.csv'
            # Delete the csv file created during super().__init__()
            if os.path.exists(csv_file):
                os.remove(csv_file)

            client_num = Config().clients.total_clients
            silo_num = Config().algorithm.cross_silo.total_silos
            epoch_num = Config().trainer.epochs
            edge_agg_num = Config().algorithm.cross_silo.rounds
            self.result_dir = f'{result_dir}/{client_num}_{silo_num}_{epoch_num}_{edge_agg_num}/'

        if Config().is_edge_server():
            # An edge client waits for the event that a certain number of
            # aggregations are completed
            self.model_aggregated = asyncio.Event()

            # An edge client waits for the event that a new global round begins
            # before starting the first round of local aggregation
            self.new_global_round_begins = asyncio.Event()

            # Compute the number of clients in each silo for edge servers
            self.total_clients = int(self.total_clients /
                                     Config().algorithm.cross_silo.total_silos)
            self.clients_per_round = int(
                self.clients_per_round /
                Config().algorithm.cross_silo.total_silos)
            logging.info(
                "[Edge server #%s] Started training with %s clients and %s per round.",
                Config().args.id, self.total_clients, self.clients_per_round)

            if hasattr(Config(), 'results'):
                result_csv_file = f'{self.result_dir}result_{Config().args.id}.csv'
                csv_processor.initialize_csv(result_csv_file,
                                             self.recorded_items,
                                             self.result_dir)

        # Compute the number of clients for the central server
        if Config().is_central_server():
            self.clients_per_round = Config().algorithm.cross_silo.total_silos
            self.total_clients = self.clients_per_round

            logging.info(
                "The central server starts training with %s edge servers.",
                self.total_clients)

            if hasattr(Config(), 'results'):
                recorded_items = self.recorded_items
                # For central server. 'round' is 'global_round'
                if 'global_round' in recorded_items:
                    recorded_items.remove('global_round')

                result_csv_file = f'{self.result_dir}result.csv'
                csv_processor.initialize_csv(result_csv_file, recorded_items,
                                             self.result_dir)

    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        if Config().args.id:
            logging.info("Configuring edge server #%s as a %s server.",
                         Config().args.id,
                         Config().algorithm.type)
            logging.info("Training with %s local aggregation rounds.",
                         Config().algorithm.cross_silo.rounds)
            self.load_model()

        else:
            super().configure()

    async def customize_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        if Config().is_central_server():
            server_response['current_global_round'] = self.current_round
        return server_response

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        if hasattr(Config(), 'results'):

            new_row = []
            for item in self.recorded_items:
                item_value = {
                    'global_round':
                    self.current_global_round,
                    'round':
                    self.current_round,
                    'accuracy':
                    self.accuracy * 100,
                    'edge_agg_num':
                    Config().algorithm.cross_silo.rounds,
                    'training_time':
                    max([report.training_time for report in self.reports]),
                    'round_time':
                    time.time() - self.round_start_time
                }[item]
                new_row.append(item_value)

            if Config().is_edge_server():
                result_csv_file = f'{self.result_dir}result_{Config().args.id}.csv'
            else:
                result_csv_file = f'{self.result_dir}result.csv'

            csv_processor.write_csv(result_csv_file, new_row)

        if Config().is_edge_server():
            # When a certain number of aggregations are completed, an edge client
            # needs to be signaled to send a report to the central server
            if self.current_round == Config().algorithm.cross_silo.rounds:
                logging.info(
                    '[Server #%d] Completed %s rounds of local aggregation.',
                    os.getpid(),
                    Config().algorithm.cross_silo.rounds)
                self.model_aggregated.set()

                self.current_round = 0
                self.new_global_round_begins.clear()
                # Wait until a new global round begins
                # to avoid selecting clients before a new global round begins
                await self.new_global_round_begins.wait()
