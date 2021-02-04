"""
Base class for trainers.
"""

from abc import ABC, abstractmethod
import time
import os
from config import Config


class Trainer(ABC):
    """Base class for all the trainers."""
    def __init__(self, client_id):
        self.device = Config().device()
        self.client_id = client_id
        Config().cursor.execute(
            "CREATE TABLE IF NOT EXISTS trainers (pid int)")

    def start_training(self):
        """Add to the list of running trainers if max_concurrency has not yet
        been reached."""
        with Config().sql_connection:
            Config().cursor.execute("SELECT COUNT(*) FROM trainers")
            trainer_count = Config().cursor.fetchone()[0]

            while trainer_count >= Config().trainer.max_concurrency:
                time.sleep(2)
                Config().cursor.execute("SELECT COUNT(*) FROM trainers")
                trainer_count = Config().cursor.fetchone()[0]

            Config().cursor.execute("INSERT INTO trainers VALUES (?)",
                                    (self.client_id, ))

    def pause_training(self):
        """Remove from the list of running trainers."""
        with Config().sql_connection:
            Config().cursor.execute("DELETE FROM trainers WHERE pid = (?)",
                                    (self.client_id, ))

        model_type = Config().trainer.model
        model_dir = Config().model_dir
        model_file = f'{model_dir}{model_type}_{self.client_id}_{Config().experiment_id}.pth'
        accuracy_file = f'{model_dir}{model_type}_{self.client_id}_{Config().experiment_id}.acc'
        c_file = f"{model_type}_c_plus_{self.client_id}_{Config().experiment_id}.pth"

        if os.path.exists(model_file):
            os.remove(model_file)

        if os.path.exists(accuracy_file):
            os.remove(accuracy_file)

        if os.path.exists(c_file):
            os.remove(c_file)

    def stop_training(self):
        """ Remove the trainers table after all training concluded."""
        Config().cursor.execute("DROP TABLE trainers")
        Config().sql_connection.close()

    @abstractmethod
    def extract_weights(self):
        """Extract weights from a model passed in as a parameter."""

    @abstractmethod
    def load_weights(self, weights):
        """Load the model weights passed in as a parameter."""

    @abstractmethod
    def train(self, trainset, cut_layer=None):
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        cut_layer (optional): The layer which training should start from.
        """

    @abstractmethod
    def test(self, testset):
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        cut_layer (optional): The layer which testing should start from.
        """
