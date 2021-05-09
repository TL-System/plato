"""
Base class for trainers.
"""

import os
import time
from abc import ABC, abstractmethod
from contextlib import closing

from config import Config


class Trainer(ABC):
    """Base class for all the trainers."""
    def __init__(self):
        self.device = Config().device()
        self.client_id = 0

    def set_client_id(self, client_id):
        """Setting the client ID and initialize the shared database table for controlling
           maximum concurrency with respect to the number of training clients.
        """
        self.client_id = client_id

        with Config().sql_connection:
            with closing(Config().sql_connection.cursor()) as cursor:
                cursor.execute(
                    "CREATE TABLE IF NOT EXISTS trainers (run_id int)")

    @staticmethod
    def save_accuracy(accuracy, filename=None):
        """Saving the test accuracy to a file."""
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if filename is not None:
            accuracy_path = f"{model_dir}{filename}"
        else:
            accuracy_path = f'{model_dir}{model_name}.acc'

        with open(accuracy_path, 'w') as file:
            file.write(str(accuracy))

    @staticmethod
    def load_accuracy(filename=None):
        """Loading the test accuracy from a file."""
        model_dir = Config().params['model_dir']
        model_name = Config().trainer.model_name

        if filename is not None:
            accuracy_path = f"{model_dir}{filename}"
        else:
            accuracy_path = f'{model_dir}{model_name}.acc'

        with open(accuracy_path, 'r') as file:
            accuracy = float(file.read())

        return accuracy

    def start_training(self):
        """Add to the list of running trainers if max_concurrency has not yet
        been reached."""
        with Config().sql_connection:
            with closing(Config().sql_connection.cursor()) as cursor:
                cursor.execute("SELECT COUNT(*) FROM trainers")
                trainer_count = cursor.fetchone()[0]

        while trainer_count >= Config().trainer.max_concurrency:
            time.sleep(self.client_id)
            with Config().sql_connection:
                with closing(Config().sql_connection.cursor()) as cursor:
                    cursor.execute("SELECT COUNT(*) FROM trainers")
                    trainer_count = cursor.fetchone()[0]

        with Config().sql_connection:
            with closing(Config().sql_connection.cursor()) as cursor:
                cursor.execute("INSERT INTO trainers VALUES (?)",
                               (self.client_id, ))

    def pause_training(self):
        """Remove from the list of running trainers."""
        with Config().sql_connection:
            with closing(Config().sql_connection.cursor()) as cursor:
                cursor.execute("DELETE FROM trainers WHERE run_id = (?)",
                               (self.client_id, ))

        model_name = Config().trainer.model_name
        model_dir = Config().params['model_dir']
        model_file = f"{model_dir}{model_name}_{self.client_id}_{Config().params['run_id']}.pth"
        accuracy_file = f"{model_dir}{model_name}_{self.client_id}_{Config().params['run_id']}.acc"

        if os.path.exists(model_file):
            os.remove(model_file)

        if os.path.exists(accuracy_file):
            os.remove(accuracy_file)

    def stop_training(self):
        """ Remove the trainers table after all training concluded."""
        with Config().sql_connection:
            with closing(Config().sql_connection.cursor()) as cursor:
                cursor.execute("DROP TABLE trainers")
        Config().sql_connection.close()

    @abstractmethod
    def train(self, trainset, sampler, cut_layer=None):
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.
        """

    @abstractmethod
    def test(self, testset):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        cut_layer (optional): The layer which testing should start from.
        """
