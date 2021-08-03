"""
Base class for trainers.
"""

import os
import random
import sqlite3
import time
from abc import ABC, abstractmethod
from typing import Tuple

from plato.config import Config


class Trainer(ABC):
    """Base class for all the trainers."""
    def __init__(self):
        self.device = Config().device()
        self.client_id = 0

    @staticmethod
    def run_sql_statement(statement: str, params: tuple = None):
        """ Run a particular command with a SQLite database connection. """
        while True:
            try:
                with Config().sql_connection:
                    if params is None:
                        Config().cursor.execute(statement)
                    else:
                        Config().cursor.execute(statement, params)

                    return_value = Config().cursor.fetchone()
                    if return_value is not None:
                        return return_value[0]
                break
            except sqlite3.OperationalError:
                time.sleep(random.random())

    def set_client_id(self, client_id):
        """ Setting the client ID and initialize the shared database table for controlling
            the maximum concurrency with respect to the number of training clients.
        """
        self.client_id = client_id
        Trainer.run_sql_statement(
            "CREATE TABLE IF NOT EXISTS trainers (run_id int)")

    @abstractmethod
    def save_model(self, filename=None):
        """Saving the model to a file. """
        raise "save_model() not implemented."

    @abstractmethod
    def load_model(self, filename=None):
        """Loading pre-trained model weights from a file. """
        raise "load_model() not implemented."

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
        time.sleep(random.random())
        trainer_count = Trainer.run_sql_statement(
            "SELECT COUNT(*) FROM trainers")

        while trainer_count >= Config().trainer.max_concurrency:
            time.sleep(random.random())
            trainer_count = Trainer.run_sql_statement(
                "SELECT COUNT(*) FROM trainers")

        Trainer.run_sql_statement("INSERT INTO trainers VALUES (?)",
                                  (self.client_id, ))

    def pause_training(self):
        """Remove from the list of running trainers."""
        Trainer.run_sql_statement("DELETE FROM trainers WHERE run_id = (?)",
                                  (self.client_id, ))

        model_name = Config().trainer.model_name
        model_dir = Config().params['model_dir']
        model_file = f"{model_dir}{model_name}_{self.client_id}_{Config().params['run_id']}.pth"
        accuracy_file = f"{model_dir}{model_name}_{self.client_id}_{Config().params['run_id']}.acc"

        if os.path.exists(model_file):
            os.remove(model_file)

        if os.path.exists(accuracy_file):
            os.remove(accuracy_file)

    @abstractmethod
    def train(self, trainset, sampler, cut_layer=None) -> Tuple[bool, float]:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        cut_layer (optional): The layer which training should start from.

        Returns:
        bool: Whether training was successfully completed.
        float: The training time.
        """

    @abstractmethod
    def test(self, testset) -> float:
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """

    @abstractmethod
    async def server_test(self, testset):
        """Testing the model on the server using the provided test dataset.

        Arguments:
        testset: The test dataset.
        """
