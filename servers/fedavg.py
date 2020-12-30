"""
A simple federated learning server using federated averaging.
"""

import logging
import time
import os
import random
import asyncio
import torch

import models.registry as models_registry
from datasets import registry as datasets_registry
from training import trainer
from servers import Server
from config import Config
from utils import csv_processor
import utils.plot_figures as plot_figures


class FedAvgServer(Server):
    """Federated learning server using federated averaging."""
    def __init__(self):
        super().__init__()
        self.testset = None
        self.model = None
        self.selected_clients = None
        self.total_samples = 0

        # starting time of a gloabl training round
        self.round_start_time = 0

        self.total_clients = Config().clients.total_clients
        self.clients_per_round = Config().clients.per_round

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
        else:
            # Compute the number of clients for the central server
            if Config().is_central_server():
                self.clients_per_round = Config().cross_silo.total_silos
                self.total_clients = self.clients_per_round

                logging.info(
                    "The central server starts training with %s edge servers...",
                    self.total_clients)
            else:
                self.clients_per_round = Config().clients.per_round
                logging.info(
                    "Started training on %s clients and %s per round...",
                    self.total_clients, self.clients_per_round)

        if Config().results:
            recorded_items = Config().results.types
            self.recorded_items = [
                x.strip() for x in recorded_items.split(',')
            ]
            self.result_csv_file = self.result_dir + 'result.csv'
            csv_processor.initialize_csv(self.result_csv_file,
                                         self.recorded_items, self.result_dir)

        random.seed()

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

        if not Config().is_edge_server():
            self.prepare_load_client_data()

    def load_test_data(self):
        """Loading the test dataset."""
        if not Config().clients.do_test:
            dataset = datasets_registry.get()
            self.testset = dataset.get_test_set()

    def load_model(self):
        """Setting up the global model to be trained via federated learning."""

        model_type = Config().training.model
        logging.info('Model: %s', model_type)

        self.model = models_registry.get(model_type)

    def prepare_load_client_data(self):
        """Preparing for loading data on clients."""
        dataset = datasets_registry.get()

        logging.info('Dataset size: %s', dataset.num_train_examples())
        logging.info('Number of classes: %s', dataset.num_classes())

        assert Config().data.divider in ('iid', 'bias', 'shard')
        logging.info('Data distribution: %s', Config().data.divider)

        num_clients = Config().clients.total_clients
        logging.info('Total number of clients: %s\n', num_clients)

    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""
        self.round_start_time = time.time()

        # Select clients randomly
        assert self.clients_per_round <= len(self.clients)
        self.selected_clients = random.sample(list(self.clients),
                                              self.clients_per_round)

    def aggregate_weights(self, reports):
        """Aggregate the reported weight updates from the selected clients."""
        return self.federated_averaging(reports)

    def extract_client_updates(self, reports):
        """Extract the model weight updates from a client's report."""
        # Extract baseline model weights
        baseline_weights = trainer.extract_weights(self.model)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, current_weight) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = current_weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def federated_averaging(self, reports):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract updates from reports
        updates = self.extract_client_updates(reports)

        # Extract total number of samples
        self.total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size()) for __, x in updates[0]]

        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            for j, (__, delta) in enumerate(update):
                # Use weighted average by the number of samples
                avg_update[j] += delta * (num_samples / self.total_samples)

        # Extract baseline model weights
        baseline_weights = trainer.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights

    def process_reports(self):
        """Process the client reports by aggregating their weights."""
        updated_weights = self.aggregate_weights(self.reports)
        trainer.load_weights(self.model, updated_weights)

        # Testing the global model accuracy
        if Config().clients.do_test:
            # Compute the average accuracy from client reports
            accuracy = self.accuracy_averaging(self.reports)
            logging.info(
                '[Server {:d}] Average client accuracy: {:.2f}%.'.format(
                    os.getpid(), 100 * accuracy))
        else:
            # Test the updated model directly at the server
            accuracy = trainer.test(self.model, self.testset,
                                    Config().training.batch_size)
            logging.info('Global model accuracy: {:.2f}%\n'.format(100 *
                                                                   accuracy))

        return accuracy

    async def wrap_up_one_round(self):
        """Wrapping up when one round of training is done."""

        # Write results into a CSV file
        if Config().results:
            if not Config().is_edge_server():
                new_row = [self.current_round]
                for item in self.recorded_items:
                    item_value = {
                        'accuracy': self.accuracy * 100,
                        'training_time': time.time() - self.round_start_time,
                        'edge_agg_num': Config().cross_silo.rounds
                    }[item]
                    new_row.append(item_value)
                csv_processor.write_csv(self.result_csv_file, new_row)

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
                # Wait until a new global round begins
                # to avoid selecting clients before a new global round begins
                await self.new_global_round_begins.wait()
                self.new_global_round_begins.clear()

    async def wrap_up(self):
        """Wrapping up when the entire training is done."""
        if Config().results:
            if Config().results.plot:
                plot_figures.plot_figures_from_dict(self.result_csv_file,
                                                    self.result_dir)

        await super().wrap_up()

    @staticmethod
    def accuracy_averaging(reports):
        """Compute the average accuracy across clients."""
        # Get total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        accuracy = 0
        for report in reports:
            accuracy += report.accuracy * (report.num_samples / total_samples)

        return accuracy

    @staticmethod
    def save_model(model_to_save, path):
        """Save the model in a file."""
        path += '/global_model'
        torch.save(model_to_save.state_dict(), path)
        logging.info('Saved the global model: %s', path)
