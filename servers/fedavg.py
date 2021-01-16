"""
A simple federated learning server using federated averaging.
"""

import logging
import time
import os
import random
import torch

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
        logging.info("Started training on %s clients and %s per round...",
                     self.total_clients, self.clients_per_round)

        # starting time of a gloabl training round
        self.round_start_time = 0

        if Config().results:
            recorded_items = Config().results.types
            self.recorded_items = ['round'] + [
                x.strip() for x in recorded_items.split(',')
            ]
            # Directory of results (figures etc.)
            result_dir = f'./results/{Config().trainer.dataset}/{Config().trainer.model}'
            result_dir += f'/{Config().server.type}/'
            result_csv_file = result_dir + 'result.csv'
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         result_dir)

        random.seed()

    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        logging.info('Configuring the %s server...', Config().server.type)

        total_rounds = Config().trainer.rounds
        target_accuracy = Config().trainer.target_accuracy

        if target_accuracy:
            logging.info('Training: %s rounds or %s%% accuracy\n',
                         total_rounds, 100 * target_accuracy)
        else:
            logging.info('Training: %s rounds\n', total_rounds)

        self.load_test_data()
        self.load_model()

    def load_test_data(self):
        """Loading the test dataset."""
        if not Config().clients.do_test:
            dataset = datasets_registry.get()
            self.testset = dataset.get_test_set()

    def load_model(self):
        """Setting up the global model to be trained via federated learning."""

        model_type = Config().trainer.model
        logging.info('Model: %s', model_type)

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

    async def customize_server_response(self, server_response):
        """Wrap up generating the server response with any additional information."""
        server_response['first_communication_start_time'] = time.time()
        return server_response

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
        baseline_weights = self.trainer.extract_weights()

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

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
                '[Server {:d}] Average client accuracy: {:.2f}%.'.format(
                    os.getpid(), 100 * self.accuracy))
        else:
            # Test the updated model directly at the server
            self.accuracy = self.trainer.test(self.testset,
                                              Config().trainer.batch_size)
            logging.info('Global model accuracy: {:.2f}%\n'.format(
                100 * self.accuracy))

        await self.wrap_up_processing_reports()

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""

        if Config().results:
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

            result_dir = f'./results/{Config().trainer.dataset}/{Config().trainer.model}/{Config().server.type}/'
            result_csv_file = result_dir + 'result.csv'

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
