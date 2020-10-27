"""
The base class for federated learning servers.
"""

from abc import abstractmethod
import logging

class Server():
    """The base class for federated learning servers."""

    def __init__(self, config):
        self.config = config
        self.dataset_type = config.general.dataset
        self.data_path = '{}/{}'.format(config.general.data_path, config.general.dataset)


    def run(self):
        """Perform consecutive rounds of federated learning training."""
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

    @abstractmethod
    def round(self):
        """
        Selecting some clients to participate in the current round,
        and run them for one round.
        """
        pass
