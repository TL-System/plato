"""
A simple federated learning server using federated averaging.
"""

import logging
import random
from threading import Thread
import torch

import client
from models import model
from utils import dists
from utils import load_data

class Server:
    """ Federated learning server using federated averaging. """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_path = '{}/{}'.format(config.general.model_path, config.general.model)
        self.loader = None
        self.clients = None # selected clients in a round


    def boot(self):
        """
        Booting the federated learning server by setting up the data, model, and
        creating the clients.
        """

        config = self.config
        total_clients = config.clients.total

        logging.info('Booting the %s server...', config.general.server)

        # Setting up the federated learning training workload
        self.load_data()
        self.load_model()
        self.make_clients(total_clients)


    def load_data(self):
        """ Generating data and loading them onto the clients. """

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
            'iid': load_data.Loader,
            'bias': load_data.BiasLoader,
            'shard': load_data.ShardLoader
        }[config.loader](config, generator)

        logging.info('Data distribution: %s', config.loader)


    def load_model(self):
        """ Setting up the global model to be trained via federated learning. """
        model_type = self.config.general.model
        logging.info('Model: %s', model_type)

        self.model = model.Net()
        self.save_model(self.model, self.model_path)


    def make_clients(self, num_clients):
        """ Generate the clients for federated learning. """

        iid = self.config.data.iid
        labels = self.loader.labels
        loader = self.config.loader
        loading = self.config.data.loading

        if not iid:  # Create a non-IID distribution for label preferences
            dist = {
                "uniform": dists.uniform,
                "normal": dists.normal
            }[self.config.clients.label_distribution](num_clients, len(labels))
            random.shuffle(dist)  # Shuffle the distribution

        # Creating emulated clients
        clients = []

        for client_id in range(num_clients):
            # Creating a new client
            new_client = client.Client(client_id)

            if not iid: # Configure this client for non-IID data
                if self.config.data.bias:
                    # Bias data partitions
                    bias = self.config.data.bias
                    # Choose weighted random preference
                    pref = random.choices(labels, dist)[0]

                    # Assign (preference, bias) configuration to the client
                    new_client.set_bias(pref, bias)

                elif self.config.data.shard:
                    # Shard data partitions
                    shard = self.config.data.shard

                    # Assign shard configuration to the client
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
            for next_client in clients:
                self.set_client_data(next_client)

        self.clients = clients


    def run(self):
        """ Run the federated learning training workload. """
        rounds = self.config.general.rounds
        target_accuracy = self.config.general.target_accuracy

        if target_accuracy:
            logging.info('Training: %s rounds or %s%% accuracy\n',
                rounds, 100 * target_accuracy)
        else:
            logging.info('Training: %s rounds\n', rounds)

        # Perform rounds of federated learning
        for current_round in range(1, rounds + 1):
            logging.info('**** Round %s/%s ****', current_round, rounds)

            # Run the federated learning round
            accuracy = self.round()

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break


    def round(self):
        """
        Selecting some clients to participate in the current round,
        and run them for one round.
        """
        sample_clients = self.select_clients()

        # Configure sample clients
        self.configure_clients(sample_clients)

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]

        for current_thread in threads:
            current_thread.start()

        for current_thread in threads:
            current_thread.join()

        # Receiving client updates
        reports = self.receive_reports(sample_clients)

        # Aggregating weight updates from the selected clients
        logging.info('Aggregating weight updates...')
        updated_weights = self.aggregate_weights(reports)

        # Load updated weights
        model.load_weights(self.model, updated_weights)

        # Save the updated global model
        self.save_model(self.model, self.model_path)

        # Test the global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(reports)
            logging.info('Average client accuracy: {:.2f}%\n'.format(100 * accuracy))
        else: # Test the updated model on the server
            testset = self.loader.get_testset()
            batch_size = self.config.general.batch_size
            testloader = model.get_testloader(testset, batch_size)
            accuracy = model.test(self.model, testloader)
            logging.info('Global model accuracy: {:.2f}%\n'.format(100 * accuracy))

        return accuracy


    # Federated learning phases

    def select_clients(self):
        """ Select devices to participate in round. """
        clients_per_round = self.config.clients.per_round

        # Select clients randomly
        sample_clients = list(random.sample(self.clients, clients_per_round))

        return sample_clients


    def configure_clients(self, sample_clients):
        """ Configure the data distribution across clients. """
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
            # Create shards if applicable
            if loader_type == 'shard':
                self.loader.create_shards()

        # Configure selected clients for federated learning task
        for selected_client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(selected_client)  # Send data partition to client

            # Extract config for client
            config = self.config

            # Continue configuraion on client
            selected_client.configure(config)


    def receive_reports(self, sample_clients):
        """ Recieve the reports from selected clients. """

        reports = [client.get_report() for client in sample_clients]

        logging.info('Reports recieved: %s', len(reports))
        assert len(reports) == len(sample_clients)

        return reports


    def aggregate_weights(self, reports):
        """ Aggregate the reported weight updates from the selected clients. """
        return self.federated_averaging(reports)


    def extract_client_updates(self, reports):
        """ Extract the model weight updates from a client's report. """

        # Extract baseline model weights
        baseline_weights = model.extract_weights(self.model)

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
        """ Aggregate weight updates from the clients using federated averaging. """

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


    @staticmethod
    def accuracy_averaging(reports):
        """ Compute the average accuracy across clients. """

        # Get total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        accuracy = 0
        for report in reports:
            accuracy += report.accuracy * (report.num_samples / total_samples)

        return accuracy

    def set_client_data(self, current_client):
        """ set the data for a client. """

        loader = self.config.loader

        # Get data partition size
        if loader != 'shard':
            if self.config.data.partition_size:
                partition_size = self.config.data.partition_size

        # Extract data partition for client
        if loader == 'iid':
            data = self.loader.get_partition(partition_size)
        elif loader == 'bias':
            data = self.loader.get_partition(partition_size, current_client.pref)
        elif loader == 'shard':
            data = self.loader.get_partition()
        else:
            logging.critical('Unknown data loader type')

        # Send data to client
        current_client.set_data(data, self.config)


    @staticmethod
    def save_model(model_to_save, path):
        """ Save the model in a file. """
        path += '/global_model'
        torch.save(model_to_save.state_dict(), path)
        logging.info('Saved the global model: %s', path)
