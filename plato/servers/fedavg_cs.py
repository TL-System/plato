"""
A cross-silo federated learning server using federated averaging, as either edge or central servers.
"""

import asyncio
import logging
import os
import random
import numpy as np

from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import registry as samplers_registry
from plato.servers import fedavg
from plato.utils import csv_processor


class Server(fedavg.Server):
    """Cross-silo federated learning server using federated averaging."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model=model, algorithm=algorithm, trainer=trainer)

        self.current_global_round = 0
        self.average_accuracy = 0

        if Config().is_edge_server():
            # An edge client waits for the event that a certain number of
            # aggregations are completed
            self.model_aggregated = asyncio.Event()

            # An edge client waits for the event that a new global round begins
            # before starting the first round of local aggregation
            self.new_global_round_begins = asyncio.Event()

            # Compute the number of clients in each silo for edge servers
            launched_clients = Config().clients.total_clients
            if hasattr(Config().clients,
                       'simulation') and Config().clients.simulation:
                launched_clients = Config().clients.per_round

            self.total_clients = [
                len(i) for i in np.array_split(np.arange(launched_clients),
                                               Config().algorithm.total_silos)
            ][Config().args.id - launched_clients - 1]

            self.clients_per_round = [
                len(i)
                for i in np.array_split(np.arange(Config().clients.per_round),
                                        Config().algorithm.total_silos)
            ][Config().args.id - launched_clients - 1]

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

            self.load_trainer()
            self.trainer.set_client_id(Config().args.id)

            # Prepares this server for processors that processes outbound and inbound
            # data payloads
            self.outbound_processor, self.inbound_processor = processor_registry.get(
                "Server", server_id=os.getpid(), trainer=self.trainer)

            if hasattr(Config().server,
                       'edge_do_test') and Config().server.edge_do_test:
                self.datasource = datasources_registry.get(
                    client_id=Config().args.id)
                self.testset = self.datasource.get_test_set()

                if hasattr(Config().data, 'edge_testset_sampler'):
                    # Set the sampler for test set
                    self.testset_sampler = samplers_registry.get(
                        self.datasource, Config().args.id, testing='edge')
                elif hasattr(Config().data, 'testset_size'):
                    # Set the Random Sampler for test set
                    from torch.utils.data import SubsetRandomSampler

                    all_inclusive = range(len(self.datasource.get_test_set()))
                    test_samples = random.sample(all_inclusive,
                                                 Config().data.testset_size)
                    self.testset_sampler = SubsetRandomSampler(test_samples)

            if hasattr(Config(), 'results'):
                result_dir = Config().params['result_dir']
                result_csv_file = f'{result_dir}/edge_{os.getpid()}.csv'
                csv_processor.initialize_csv(result_csv_file,
                                             self.recorded_items, result_dir)

        else:
            super().configure()
            if hasattr(Config().server, 'do_test') and Config().server.do_test:
                self.datasource = datasources_registry.get(client_id=0)
                self.testset = self.datasource.get_test_set()

                if hasattr(Config().data, 'testset_size'):
                    from torch.utils.data import SubsetRandomSampler

                    # Set the sampler for testset
                    all_inclusive = range(len(self.datasource.get_test_set()))
                    test_samples = random.sample(all_inclusive,
                                                 Config().data.testset_size)
                    self.testset_sampler = SubsetRandomSampler(test_samples)

    async def select_clients(self):
        if Config().is_edge_server():
            if self.current_round == 0:
                # Wait until this edge server is selected by the central server
                # to avoid the edge server selects clients and clients begin training
                # before the edge server is selected
                await self.new_global_round_begins.wait()
                self.new_global_round_begins.clear()

        await super().select_clients()

    async def customize_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        if Config().is_central_server():
            server_response['current_global_round'] = self.current_round
        return server_response

    async def process_reports(self):
        """Process the client reports by aggregating their weights."""
        # To pass the client_id == 0 assertion during aggregation
        self.trainer.set_client_id(0)
        await self.aggregate_weights(self.updates)
        if Config().is_edge_server():
            self.trainer.set_client_id(Config().args.id)

        # Testing the model accuracy
        if (Config().is_edge_server() and Config().clients.do_test) or (
                Config().is_central_server()
                and hasattr(Config().server, 'edge_do_test')
                and Config().server.edge_do_test):
            # Compute the average accuracy from client reports
            self.average_accuracy = self.accuracy_averaging(self.updates)
            logging.info('[%s] Average client accuracy: %.2f%%.', self,
                         100 * self.average_accuracy)
        elif Config().is_central_server() and Config().clients.do_test:
            # Compute the average accuracy from client reports
            self.average_accuracy = self.client_accuracy_averaging()
            logging.info('[%s] Average client accuracy: %.2f%%.', self,
                         100 * self.average_accuracy)

        if Config().is_central_server() and hasattr(
                Config().server, 'do_test') and Config().server.do_test:
            # Test the updated model directly at the central server
            self.accuracy = await self.trainer.server_test(
                self.testset, self.testset_sampler)
            if hasattr(Config().trainer, 'target_perplexity'):
                logging.info('[%s] Global model perplexity: %.2f\n', self,
                             self.accuracy)
            else:
                logging.info('[%s] Global model accuracy: %.2f%%\n', self,
                             100 * self.accuracy)
        elif Config().is_edge_server() and hasattr(
                Config().server,
                'edge_do_test') and Config().server.edge_do_test:
            # Test the aggregated model directly at the edge server
            self.accuracy = self.trainer.test(self.testset,
                                              self.testset_sampler)
            if hasattr(Config().trainer, 'target_perplexity'):
                logging.info('[%s] Aggregated model perplexity: %.2f\n', self,
                             self.accuracy)
            else:
                logging.info('[%s] Aggregated model accuracy: %.2f%%\n', self,
                             100 * self.accuracy)
        else:
            self.accuracy = self.average_accuracy

        await self.wrap_up_processing_reports()

    def client_accuracy_averaging(self):
        """Compute the average accuracy across clients."""
        # Get total number of samples
        total_samples = sum(
            [report.num_samples for (report, __, __) in self.updates])

        # Perform weighted averaging
        accuracy = 0
        for (report, __, __) in self.updates:
            accuracy += report.average_accuracy * (report.num_samples /
                                                   total_samples)

        return accuracy

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        if hasattr(Config(), 'results'):
            new_row = []
            for item in self.recorded_items:
                item_value = self.get_record_items_values()[item]
                new_row.append(item_value)

            if Config().is_edge_server():
                result_csv_file = f"{Config().params['result_dir']}/edge_{os.getpid()}.csv"
            else:
                result_csv_file = f"{Config().params['result_dir']}/{os.getpid()}.csv"

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
                self.current_global_round += 1

    def get_record_items_values(self):
        """Get values will be recorded in result csv file."""
        return {
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
            'elapsed_time':
            self.wall_time - self.initial_wall_time,
            'comm_time':
            max([report.comm_time for (report, __, __) in self.updates]),
            'round_time':
            max([
                report.training_time + report.comm_time
                for (report, __, __) in self.updates
            ]),
        }

    async def wrap_up(self):
        """Wrapping up when each round of training is done."""
        if Config().is_central_server():
            await super().wrap_up()
