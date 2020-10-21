"""
A simple federated learning server using federated averaging.
"""

import logging
import sys
import random
import numpy as np
import torch
import pickle
from threading import Thread

import client
from utils import dists
from utils import load_data

class Server:
    """ Federated learning server using federated averaging. """

    def __init__(self, config):
        self.config = config


    def boot(self):
        '''
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        '''

        config = self.config
        self.model_path = '{}/{}'.format(config.general.model_path, config.general.model)
        total_clients = config.clients.total

        logging.info('Booting the %s server...', config.general.server)

        # Adding the federated learning model to the import path
        sys.path.append(self.model_path)

        # Setting up a simulated server
        self.load_data()
        self.load_model()
        self.make_clients(total_clients)


    def load_data(self):
        ''' Generating data and loading them at the clients. '''
        import model

        # Extract configurations for loaders
        config = self.config

        # Set up the data generator
        generator = model.Generator()

        # Generate the data
        data_path = config.general.data_path
        data = generator.generate(data_path)
        labels = generator.labels

        logging.info('Dataset size: %s',
            sum([len(x) for x in [data[label] for label in labels]]))
        logging.debug('Labels (%s): %s', len(labels), labels)

        # Setting up the data loader
        self.loader = {
            'basic': load_data.Loader(config, generator),
            'bias': load_data.BiasLoader(config, generator),
            'shard': load_data.ShardLoader(config, generator)
        }[config.loader]

        logging.info('Loader: %s, IID: %s', config.loader, config.data.IID)


    def load_model(self):
        import model

        config = self.config

        model_type = config.general.model

        logging.info('Model: %s', model_type)

        # Set up global model
        self.model = model.Net()
        self.save_model(self.model, self.model_path)

        # Extract flattened weights (if applicable)
        if self.config.general.report_path:
            self.saved_reports = {}
            self.save_reports(0, []) # Save the initial model

    def make_clients(self, num_clients):
        import utils.dists as dists

        IID = self.config.data.IID
        labels = self.loader.labels
        loader = self.config.loader
        loading = self.config.data.loading

        if not IID:  # Create distribution for label preferences if non-IID
            dist = {
                "uniform": dists.uniform(num_clients, len(labels)),
                "normal": dists.normal(num_clients, len(labels))
            }[self.config.clients.label_distribution]
            random.shuffle(dist)  # Shuffle distribution

        # Creating emulated clients
        clients = []
        for client_id in range(num_clients):

            # Create a new client
            new_client = client.Client(client_id)

            if not IID:  # Configure clients for non-IID data
                if self.config.data.bias:
                    # Bias data partitions
                    bias = self.config.data.bias
                    # Choose weighted random preference
                    pref = random.choices(labels, dist)[0]

                    # Assign preference, bias config
                    new_client.set_bias(pref, bias)
                elif self.config.data.shard:
                    # Shard data partitions
                    shard = self.config.data.shard

                    # Assign shard config
                    new_client.set_shard(shard)

            clients.append(new_client)

        logging.info('Total number of clients: %s', len(clients))

        if loader == 'bias':
            logging.info('Label distribution: %s',
                [[client.pref for client in clients].count(label) for label in labels])

        if loading == 'static':
            if loader == 'shard': # Create data shards
                self.loader.create_shards()

            # Send data partition to all clients
            [self.set_client_data(client) for client in clients]

        self.clients = clients

    # Run federated learning
    def run(self):
        rounds = self.config.general.rounds
        target_accuracy = self.config.general.target_accuracy
        reports_path = self.config.general.report_path

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        # Perform rounds of federated learning
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

            # Run the federated learning round
            accuracy = self.round()

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))

    def round(self):
        import model

        # Select clients to participate in the round
        sample_clients = self.selection()

        # Configure sample clients
        self.configuration(sample_clients)

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client updates
        reports = self.reporting(sample_clients)

        # Perform weight aggregation
        logging.info('Aggregating updates...')
        updated_weights = self.aggregation(reports)

        # Load updated weights
        model.load_weights(self.model, updated_weights)

        # Extract flattened weights (if applicable)
        if self.config.general.report_path:
            self.save_reports(round, reports)

        # Save the updated global model
        self.save_model(self.model, self.model_path)

        # Test the global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(reports)
        else: # Test the updated model on the server
            testset = self.loader.get_testset()
            batch_size = self.config.general.batch_size
            testloader = model.test_loader(testset, batch_size)
            accuracy = model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
        return accuracy

    # Federated learning phases

    def selection(self):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round

        # Select clients randomly
        sample_clients = [client for client in random.sample(
            self.clients, clients_per_round)]

        return sample_clients

    def configuration(self, sample_clients):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
            # Create shards if applicable
            if loader_type == 'shard':
                self.loader.create_shards()

        # Configure selected clients for federated learning task
        for client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(client)  # Send data partition to client

            # Extract config for client
            config = self.config

            # Continue configuraion on client
            client.configure(config)

    def reporting(self, sample_clients):
        # Recieve reports from sample clients
        reports = [client.get_report() for client in sample_clients]

        logging.info('Reports recieved: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)

        return reports

    def aggregation(self, reports):
        return self.federated_averaging(reports)

    # Report aggregation
    def extract_client_updates(self, reports):
        import model

        # Extract baseline model weights
        baseline_weights = model.extract_weights(self.model)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, weight) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates


    def federated_averaging(self, reports):
        import model

        # Extract updates from reports
        updates = self.extract_client_updates(reports)

        # Extract total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size())
                      for _, x in updates[0]]
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            for j, (_, delta) in enumerate(update):
                # Use weighted average by number of samples
                avg_update[j] += delta * (num_samples / total_samples)

        # Extract baseline model weights
        baseline_weights = model.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights

    def accuracy_averaging(self, reports):
        # Get total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        accuracy = 0
        for report in reports:
            accuracy += report.accuracy * (report.num_samples / total_samples)

        return accuracy

    # Server operations
    @staticmethod
    def flatten_weights(weights):
        # Flatten weights into vectors
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)

    def set_client_data(self, client):
        loader = self.config.loader

        # Get data partition size
        if loader != 'shard':
            if self.config.data.partition_size:
                partition_size = self.config.data.partition_size

        # Extract data partition for client
        if loader == 'basic':
            data = self.loader.get_partition(partition_size)
        elif loader == 'bias':
            data = self.loader.get_partition(partition_size, client.pref)
        elif loader == 'shard':
            data = self.loader.get_partition()
        else:
            logging.critical('Unknown data loader type')

        # Send data to client
        client.set_data(data, self.config)

    def save_model(self, model, path):
        path += '/global_model'
        torch.save(model.state_dict(), path)
        logging.info('Saved the global model: {}'.format(path))

    def save_reports(self, round, reports):
        import model

        if reports:
            self.saved_reports['round{}'.format(round)] = [(report.client_id, 
                self.flatten_weights(report.weights)) for report in reports]

        # Extract global weights
        self.saved_reports['w{}'.format(round)] = self.flatten_weights(
            model.extract_weights(self.model))
