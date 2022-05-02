"""
A simple federated learning server using federated averaging.
"""

import asyncio
import logging
import os
import random

from plato.algorithms import registry as algorithms_registry
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.servers import base
from plato.trainers import registry as trainers_registry
from plato.utils import csv_processor


class Server(base.Server):
    """Federated learning server using federated averaging."""
    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__()

        self.model = model
        self.algorithm = algorithm
        
        self.custom_trainer = trainer
        self.trainer = None

        self.datasource = None
        self.testset = None
        self.testset_sampler = None
        self.total_samples = 0

        self.total_clients = Config().clients.total_clients
        self.clients_per_round = Config().clients.per_round

        logging.info(
            "[Server #%d] Started training on %d clients with %d per round.",
            os.getpid(), self.total_clients, self.clients_per_round)

        if hasattr(Config(), 'results'):
            recorded_items = Config().results.types
            self.recorded_items = [
                x.strip() for x in recorded_items.split(',')
            ]

    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        logging.info("[Server #%d] Configuring the server...", os.getpid())
        super().configure()

        total_rounds = Config().trainer.rounds
        target_accuracy = None
        target_perplexity = None

        if hasattr(Config().trainer, 'target_accuracy'):
            target_accuracy = Config().trainer.target_accuracy
        elif hasattr(Config().trainer, 'target_perplexity'):
            target_perplexity = Config().trainer.target_perplexity

        if target_accuracy:
            logging.info("Training: %s rounds or accuracy above %.1f%%\n",
                         total_rounds, 100 * target_accuracy)
        elif target_perplexity:
            logging.info("Training: %s rounds or perplexity below %.1f\n",
                         total_rounds, target_perplexity)
        else:
            logging.info("Training: %s rounds\n", total_rounds)

        self.load_trainer()

        # Prepares this server for processors that processes outbound and inbound
        # data payloads
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Server", server_id=os.getpid(), trainer=self.trainer)

        if not Config().clients.do_test:
            self.datasource = datasources_registry.get(client_id=0)
            self.testset = self.datasource.get_test_set()

            if hasattr(Config().data, 'testset_size'):
                # Set the sampler for testset
                import torch

                if hasattr(Config().server, "random_seed"):
                    random_seed = Config().server.random_seed
                else:
                    random_seed = 1

                gen = torch.Generator()
                gen.manual_seed(random_seed)

                all_inclusive = range(len(self.datasource.get_test_set()))
                test_samples = random.sample(all_inclusive,
                                             Config().data.testset_size)
                self.testset_sampler = torch.utils.data.SubsetRandomSampler(
                    test_samples, generator=gen)

        # Initialize the csv file which will record results
        if hasattr(Config(), 'results'):
            result_csv_file = f"{Config().params['result_dir']}/{os.getpid()}.csv"
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         Config().params['result_dir'])

    def load_trainer(self):
        """Setting up the global model to be trained via federated learning."""
        if self.trainer is None and self.custom_trainer is None:
            self.trainer = trainers_registry.get(model=self.model)
        elif self.custom_trainer is not None:
            self.trainer = self.custom_trainer()
            self.custom_trainer = None

        self.trainer.set_client_id(0)

        if self.algorithm is None:
            self.algorithm = algorithms_registry.get(self.trainer)

    async def select_clients(self):
        await super().select_clients()

    def extract_client_updates(self, updates):
        """Extract the model weight updates from client updates."""
        weights_received = [payload for (__, payload, __) in updates]
        return self.algorithm.compute_weight_updates(weights_received)

    async def aggregate_weights(self, updates):
        """Aggregate the reported weight updates from the selected clients."""
        update = await self.federated_averaging(updates)
        updated_weights = self.algorithm.update_weights(update)
        self.algorithm.load_weights(updated_weights)

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        weights_received = self.extract_client_updates(updates)

        # Extract the total number of samples
        self.total_samples = sum(
            [report.num_samples for (report, __, __) in updates])

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            report, __, __ = updates[i]
            num_samples = report.num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update

    async def process_reports(self):
        """Process the client reports by aggregating their weights."""
        await self.aggregate_weights(self.updates)

        # Testing the global model accuracy
        if Config().clients.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info('[%s] Average client accuracy: %.2f%%.', self,
                         100 * self.accuracy)
        else:
            # Testing the updated model directly at the server
            self.accuracy = await self.trainer.server_test(
                self.testset, self.testset_sampler)

        if hasattr(Config().trainer, 'target_perplexity'):
            logging.info('[%s] Global model perplexity: %.2f\n', self,
                         self.accuracy)
        else:
            logging.info('[%s] Global model accuracy: %.2f%%\n', self,
                         100 * self.accuracy)

        await self.wrap_up_processing_reports()

    async def wrap_up_processing_reports(self):
        """ Wrap up processing the reports with any additional work. """
        if hasattr(Config(), 'results'):
            new_row = []

            for item in self.recorded_items:
                item_value = {
                    'round':
                    self.current_round,
                    'accuracy':
                    self.accuracy,
                    'elapsed_time':
                    self.wall_time - self.initial_wall_time,
                    'comm_time':
                    max([
                        report.comm_time for (report, __, __) in self.updates
                    ]),
                    'round_time':
                    max([
                        report.training_time + report.comm_time
                        for (report, __, __) in self.updates
                    ]),
                }[item]
                new_row.append(item_value)

            result_csv_file = f"{Config().params['result_dir']}/{os.getpid()}.csv"
            csv_processor.write_csv(result_csv_file, new_row)

    @staticmethod
    def accuracy_averaging(updates):
        """Compute the average accuracy across clients."""
        # Get total number of samples
        total_samples = sum(
            [report.num_samples for (report, __, __) in updates])

        # Perform weighted averaging
        accuracy = 0
        for (report, __, __) in updates:
            accuracy += report.accuracy * (report.num_samples / total_samples)

        return accuracy

    def customize_server_payload(self, payload):
        """ Customize the server payload before sending to the client. """
        return payload
