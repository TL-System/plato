"""
A cross-silo federated learning server using federated averaging, as either edge or central servers.
"""

import asyncio
import logging
import os
import time

from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.utils import csv_processor

from plato.servers import fedavg


class Server(fedavg.Server):
    """Cross-silo federated learning server using federated averaging."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=None, algorithm=None, trainer=None)

        self.current_global_round = None
        # Average accuracy from client reports
        self.average_accuracy = 0

        if Config().is_edge_server():
            # An edge client waits for the event that a certain number of
            # aggregations are completed
            self.model_aggregated = asyncio.Event()

            # An edge client waits for the event that a new global round begins
            # before starting the first round of local aggregation
            self.new_global_round_begins = asyncio.Event()

            # Compute the number of clients in each silo for edge servers
            self.total_clients = int(self.total_clients /
                                     Config().algorithm.total_silos)
            self.clients_per_round = int(self.clients_per_round /
                                         Config().algorithm.total_silos)
            logging.info(
                "[Edge server #%d (#%d)] Started training on %d clients with %d per round.",
                Config().args.id, os.getpid(), self.total_clients,
                self.clients_per_round)

            if hasattr(Config(), 'results'):
                self.recorded_items = ['global_round'] + self.recorded_items

        # Compute the number of clients for the central server
        if Config().is_central_server():
            self.clients_per_round = Config().algorithm.total_silos
            self.total_clients = self.clients_per_round

            logging.info(
                "The central server starts training with %s edge servers.",
                self.total_clients)

    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        if Config().is_edge_server():
            logging.info("Configuring edge server #%d as a %s server.",
                         Config().args.id,
                         Config().algorithm.type)
            logging.info("Training with %s local aggregation rounds.",
                         Config().algorithm.local_rounds)

            if hasattr(Config().server, 'do_test'):
                if not Config().clients.do_test or Config().server.do_test:
                    datasource = datasources_registry.get()
                    self.testset = datasource.get_test_set()

            self.load_trainer()

            if hasattr(Config(), 'results'):
                result_dir = Config().result_dir
                result_csv_file = f'{result_dir}/result_{Config().args.id}.csv'
                csv_processor.initialize_csv(result_csv_file,
                                             self.recorded_items, result_dir)

        else:
            super().configure()

            if hasattr(Config().server, 'do_test'):
                if Config().clients.do_test and Config().server.do_test:
                    datasource = datasources_registry.get()
                    self.testset = datasource.get_test_set()

    async def select_clients(self):
        if Config().is_edge_server():
            if self.current_round == 0:
                # Wait until this edge server is selected by the central server
                # to avoid the edge server selects clients and clients begin training
                # before the edge server is selected
                await self.new_global_round_begins.wait()

        await super().select_clients()

    async def customize_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        if Config().is_central_server():
            server_response['current_global_round'] = self.current_round
        return server_response

    async def process_reports(self):
        """Process the client reports by aggregating their weights."""
        await self.aggregate_weights(self.updates)

        # Testing the global model accuracy
        if Config().clients.do_test:
            # Compute the average accuracy from client reports
            self.average_accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                '[Server #{:d}] Average client accuracy: {:.2f}%.'.format(
                    os.getpid(), 100 * self.average_accuracy))

        if hasattr(Config().server, 'do_test'):
            if not Config().clients.do_test or Config().server.do_test:
                # Test the updated model directly at the server
                self.accuracy = self.trainer.test(self.testset)
                logging.info(
                    '[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
                        os.getpid(), 100 * self.accuracy))
        else:
            self.accuracy = self.average_accuracy

        await self.wrap_up_processing_reports()

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
                    'average_accuracy':
                    self.average_accuracy * 100,
                    'edge_agg_num':
                    Config().algorithm.local_rounds,
                    'local_epoch_num':
                    Config().trainer.epochs,
                    'training_time':
                    max([
                        report.training_time for (report, __) in self.updates
                    ]),
                    'round_time':
                    time.perf_counter() - self.round_start_time
                }[item]
                new_row.append(item_value)

            if Config().is_edge_server():
                result_csv_file = f'{Config().result_dir}result_{Config().args.id}.csv'
            else:
                result_csv_file = f'{Config().result_dir}result.csv'

            csv_processor.write_csv(result_csv_file, new_row)

        if Config().is_edge_server():
            # When a certain number of aggregations are completed, an edge client
            # needs to be signaled to send a report to the central server
            if self.current_round == Config().algorithm.local_rounds:
                logging.info(
                    '[Server #%d] Completed %s rounds of local aggregation.',
                    os.getpid(),
                    Config().algorithm.local_rounds)
                self.model_aggregated.set()

                self.current_round = 0
                self.new_global_round_begins.clear()
                # Wait until a new global round begins
                # to avoid selecting clients before a new global round begins
                await self.new_global_round_begins.wait()

    async def wrap_up(self):
        """Wrapping up when each round of training is done."""
        if Config().is_central_server():
            await super().wrap_up()
