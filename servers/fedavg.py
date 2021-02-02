"""
A simple federated learning server using federated averaging.
"""

import logging
import time
import os
import random
from collections import OrderedDict

import models.registry as models_registry
from datasets import registry as datasets_registry
from trainers import registry as trainers_registry
from servers import Server
from config import Config
from utils import csv_processor


class FedAvgServer(Server):
    """Federated learning server using federated averaging."""
    def __init__(self):
        super().__init__()
        self.testset = None
        self.model = None
        self.selected_clients = None
        self.total_samples = 0

        self.total_clients = Config().clients.total_clients
        self.clients_per_round = Config().clients.per_round
        logging.info(
            "[Server #%d] Started training on %s clients with %s per round.",
            os.getpid(), self.total_clients, self.clients_per_round)

        # starting time of a global training round
        self.round_start_time = 0

        if hasattr(Config(), 'results'):
            recorded_items = Config().results.types
            self.recorded_items = ['round'] + [
                x.strip() for x in recorded_items.split(',')
            ]

        random.seed()

    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """

        logging.info("[Server #%s] Configuring the %s server...", os.getpid(),
                     Config().algorithm.type)

        total_rounds = Config().trainer.rounds
        target_accuracy = Config().trainer.target_accuracy

        if target_accuracy:
            logging.info("Training: %s rounds or %s%% accuracy\n",
                         total_rounds, 100 * target_accuracy)
        else:
            logging.info("Training: %s rounds\n", total_rounds)

        if not Config().clients.do_test:
            dataset = datasets_registry.get()
            self.testset = dataset.get_test_set()

        self.load_model()

        # Initialize the csv file which will record results
        if hasattr(Config(), 'results'):
            result_csv_file = Config().result_dir + 'result.csv'
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         Config().result_dir)

    def load_model(self):
        """Setting up the global model to be trained via federated learning."""

        model_type = Config().trainer.model
        logging.info("[Server #%s] Model: %s", os.getpid(), model_type)

        self.model = models_registry.get(model_type)
        self.trainer = trainers_registry.get(self.model)

    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""
        # Select clients randomly
        assert self.clients_per_round <= len(self.clients)
        self.selected_clients = random.sample(list(self.clients),
                                              self.clients_per_round)
        # starting time of a gloabl training round
        self.round_start_time = time.time()

    def aggregate_weights(self, reports):
        """Aggregate the reported weight updates from the selected clients."""
        return self.federated_averaging(reports)

    def extract_client_updates(self, reports):
        """Extract the model weight updates from a client's report."""

        # Extract weights from reports
        weights_received = [report.weights for report in reports]
        return self.trainer.compute_weight_updates(weights_received)

    def federated_averaging(self, reports):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract updates from the reports
        updates = self.extract_client_updates(reports)

        # Extract the total number of samples
        self.total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in updates[0].items()
        }

        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples

            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (num_samples / self.total_samples)

        # Extract baseline model weights
        baseline_weights = self.trainer.extract_weights()

        # Load updated weights into model
        updated_weights = OrderedDict()
        for name, weight in baseline_weights.items():
            updated_weights[name] = weight + avg_update[name]

        return updated_weights

    async def process_reports(self):
        """Process the client reports by aggregating their weights."""
        updated_weights = self.aggregate_weights(self.reports)
        self.trainer.load_weights(updated_weights)

        # Testing the global model accuracy
        if Config().clients.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.reports)
            logging.info(
                '[Server #{:d}] Average client accuracy: {:.2f}%.'.format(
                    os.getpid(), 100 * self.accuracy))
        else:
            # Test the updated model directly at the server
            self.accuracy = self.trainer.test(self.testset)
            logging.info(
                '[Server #{:d}] Global model accuracy: {:.2f}%\n'.format(
                    os.getpid(), 100 * self.accuracy))

        await self.wrap_up_processing_reports()

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""

        if hasattr(Config(), 'results'):
            new_row = []
            for item in self.recorded_items:
                item_value = {
                    'round':
                    self.current_round,
                    'accuracy':
                    self.accuracy * 100,
                    'training_time':
                    max([report.training_time for report in self.reports]),
                    'round_time':
                    time.time() - self.round_start_time
                }[item]
                new_row.append(item_value)

            result_csv_file = Config().result_dir + 'result.csv'

            csv_processor.write_csv(result_csv_file, new_row)

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
