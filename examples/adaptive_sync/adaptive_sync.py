"""
A federated learning client with support for Adaptive Synchronization Frequency.

Reference:

C. Chen, et al. "GIFT: Towards Accurate and Efficient Federated
Learning withGradient-Instructed Frequency Tuning," found in papers/.
"""
import os

os.environ['config_file'] = 'adaptive_sync_MNIST_lenet5.yml'

from plato.servers import fedavg
from plato.trainers import basic

import adaptive_sync_algorithm
import adaptive_sync_client


def main():
    """ A Plato federated learning training session using Adaptive Synchronization Frequency. """
    trainer = basic.Trainer()
    algorithm = adaptive_sync_algorithm.Algorithm(trainer=trainer)
    client = adaptive_sync_client.Client(algorithm=algorithm, trainer=trainer)
    server = fedavg.Server(algorithm=algorithm, trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
