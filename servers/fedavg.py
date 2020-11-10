"""
A simple federated learning server using federated averaging.
"""

import logging
import random
import torch

import models.registry as models_registry
from clients import SimpleClient
import datasets
from datasets import registry as datasets_registry
from dividers import iid, biased, sharded
from training import trainer
from utils import dists, executor
from servers import Server


class FedAvgServer(Server):
    """Federated learning server using federated averaging."""

    def __init__(self, config):
        super().__init__(config)
        self.loader = None


    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """
        config = self.config

        logging.info('Configuring the %s server...', config.training.server)

        self.load_model()


    def load_model(self):
        """Setting up the global model to be trained via federated learning."""
        logging.info('Dataset: %s', self.dataset_type)
        logging.info('Dataset path: %s', self.data_path)

        model_type = self.config.training.model
        logging.info('Model: %s', model_type)

        self.model = models_registry.get(model_type, self.config)


    def select_clients(self):
        """Select devices to participate in round."""
        clients_per_round = self.config.clients.per_round

        # Select clients randomly
        print(clients_per_round)
        print(len(self.clients))
        assert clients_per_round <= len(self.clients)

        selected_clients = random.sample(list(self.clients), clients_per_round)
        print(selected_clients)
        return selected_clients


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
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size())
                      for __, x in updates[0]]
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            for j, (__, delta) in enumerate(update):
                # Use weighted average by number of samples
                avg_update[j] += delta * (num_samples / total_samples)

        # Extract baseline model weights
        baseline_weights = trainer.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights


    def process_report(self):
        updated_weights = self.aggregate_weights(self.reports)
        trainer.load_weights(self.model, updated_weights)

        # Test the global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(self.reports)
            logging.info('Average client accuracy: {:.2f}%\n'.format(100 * accuracy))
        else: # Test the updated model on the server
            testset = self.loader.get_testset()
            batch_size = self.config.training.batch_size
            testloader = trainer.get_testloader(testset, batch_size)
            accuracy = trainer.test(self.model, testloader)
            logging.info('Global model accuracy: {:.2f}%\n'.format(100 * accuracy))

        return accuracy


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
