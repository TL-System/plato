"""
A basic personalized federated learning client
who performs the global learning and local learning.

At the current stage, the personalized client only
supports the data statistics saving

"""

import os
import json
import logging

from collections import Counter
from attr import has

import torch

from plato.config import Config
from plato.clients import simple


class Client(simple.Client):
    """A basic personalized federated learning client."""

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

    def perform_data_statistics(self, dataset, dataset_sampler):
        """ Record the data statistics. """
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            sampler=dataset_sampler.get())
        data_labels = []
        for _, label in data_loader:
            data_labels.extend(label.tolist())
        num_samples = len(data_labels)
        labels_sample_count = Counter(data_labels)
        return labels_sample_count, num_samples

    def save_data_statistics(self):

        result_path = Config().params['result_path']

        save_location = os.path.join(result_path,
                                     "client_" + str(self.client_id))
        os.makedirs(save_location, exist_ok=True)
        filename = f"client_{self.client_id}_data_statistics.json"
        save_file_path = os.path.join(save_location, filename)

        if not os.path.exists(save_file_path):

            train_data_sta, train_count = self.perform_data_statistics(
                self.trainset, self.sampler)
            test_data_sta, test_count = self.perform_data_statistics(
                self.testset, self.testset_sampler)

            with open(save_file_path, 'w') as fp:
                json.dump(
                    {
                        "train": train_data_sta,
                        "test": test_data_sta,
                        "train_size": train_count,
                        "test_size": test_count
                    }, fp)
            logging.info(f"Saved the {self.client_id}'s local data statistics")

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        super().load_data()

        if hasattr(Config().clients, "do_data_tranform_logging") and Config(
        ).clients.do_data_tranform_logging:
            self.save_data_statistics()
