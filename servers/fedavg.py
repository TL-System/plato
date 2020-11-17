"""
A simple federated learning server using federated averaging.
"""

import logging
import random
import torch

import models.registry as models_registry
from datasets import registry as datasets_registry
from training import trainer
from servers import Server
from config import Config


class FedAvgServer(Server):
    """Federated learning server using federated averaging."""

    def __init__(self):
        super().__init__()
        self.testset = None
        self.model = None
        self.selected_clients = None
        self.rounds = 0
        self.total_samples = 0
        random.seed()


    def configure(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """

        logging.info('Configuring the %s server...', Config().server.type)

        total_rounds = Config().training.rounds
        target_accuracy = Config().training.target_accuracy

        if target_accuracy:
            logging.info('Training: %s rounds or %s%% accuracy\n',
                total_rounds, 100 * target_accuracy)
        else:
            logging.info('Training: %s rounds\n', total_rounds)

        logging.info("Starting training on %s clients...", Config().clients.per_round)

        self.load_test_data()
        self.load_model()


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


    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""
        clients_per_round = Config().clients.per_round

        # Select clients randomly
        assert clients_per_round <= len(self.clients)
        self.selected_clients = random.sample(list(self.clients), clients_per_round)


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
        avg_update = [torch.zeros(x.size())
                      for __, x in updates[0]]

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


    def process_report(self):
        """Process the client reports by aggregating their weights."""
        updated_weights = self.aggregate_weights(self.reports)
        trainer.load_weights(self.model, updated_weights)

        # Testing the global model accuracy
        if Config().clients.do_test:
            # Compute the average accuracy from client reports
            accuracy = self.accuracy_averaging(self.reports)
            logging.info('Average client accuracy: {:.2f}%\n'.format(100 * accuracy))
        else:
            # Test the updated model directly at the server
            accuracy = trainer.test(self.model, self.testset, Config().training.batch_size)
            logging.info('Global model accuracy: {:.2f}%\n'.format(100 * accuracy))

        self.rounds += 1

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
