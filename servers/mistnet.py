"""
A federated learning server for MistNet.

Reference:

P. Wang, et al. "MistNet: Towards Private Neural Network Training with Local Differential Privacy"
"""

import logging
import time
import random
import torch
from itertools import chain

import models.registry as models_registry
from datasets import registry as datasets_registry
from trainers import registry as trainers_registry
from servers import Server
from config import Config
from utils import csv_processor


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label


class MistNetServer(Server):
    """A federated learning server for MistNet."""
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
        logging.info("Started training on %s clients and %s per round...",
                     self.total_clients, self.clients_per_round)

        if Config().results:
            recorded_items = Config().results.types
            self.recorded_items = [
                x.strip() for x in recorded_items.split(',')
            ]
            # Directory of results (figures etc.)
            result_dir = f'./results/{Config().training.dataset}/{Config().training.model}/'
            result_csv_file = result_dir + 'result.csv'
            csv_processor.initialize_csv(result_csv_file, self.recorded_items,
                                         result_dir)

        random.seed()

    def configure(self):
        """
        Booting the MistNet server by setting up the data, model, and
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

        self.load_test_data()
        self.load_model()

    def load_test_data(self):
        """Loading the test dataset."""
        dataset = datasets_registry.get()
        self.testset = dataset.get_test_set()

    def load_model(self):
        """Setting up a pre-trained model to be loaded on the clients."""
        model_type = Config().training.model
        logging.info('Model: %s', model_type)

        # Loading the model for server-side training
        self.model = models_registry.get(model_type)
        self.trainer = trainers_registry.get(self.model)

        # Loading pre-trained weights for the model
        model_path = f'./models/pretrained/{model_type}.pth'
        self.model.load_state_dict(torch.load(model_path))

    def choose_clients(self):
        """Choose a subset of the clients to participate in each round."""
        self.round_start_time = time.time()

        # Select clients randomly
        assert self.clients_per_round <= len(self.clients)
        self.selected_clients = random.sample(list(self.clients),
                                              self.clients_per_round)

    async def process_reports(self):
        """Process the features extracted by the client and perform server-side training."""
        features = [report.features for report in self.reports]

        # Faster way to deep flatten a list of lists compared to list comprehension
        feature_dataset = list(chain.from_iterable(features))

        # Traing the model using features received from the client
        self.trainer.train(FeatureDataset(feature_dataset),
                           Config().training.cut_layer)

        # Test the updated model
        self.accuracy = self.trainer.test(FeatureDataset(feature_dataset),
                                          Config().training.batch_size,
                                          Config().training.cut_layer)
        logging.info('Global model accuracy: {:.2f}%\n'.format(100 *
                                                               self.accuracy))

        await self.wrap_up_processing_reports()

    async def wrap_up_processing_reports(self):
        """Wrap up processing the reports with any additional work."""
        if Config().results:
            new_row = [self.current_round]
            for item in self.recorded_items:
                item_value = {
                    'accuracy': self.accuracy * 100,
                    'training_time': time.time() - self.round_start_time
                }[item]
                new_row.append(item_value)

            result_dir = f'./results/{Config().training.dataset}/{Config().training.model}/'
            result_csv_file = result_dir + 'result.csv'

            csv_processor.write_csv(result_csv_file, new_row)
