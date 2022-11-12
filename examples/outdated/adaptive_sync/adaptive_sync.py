"""
A federated learning client with support for Adaptive Synchronization Frequency.

Reference:

C. Chen, et al. "GIFT: Towards Accurate and Efficient Federated
Learning withGradient-Instructed Frequency Tuning," found in papers/.
"""

from plato.servers import fedavg

import adaptive_sync_algorithm
import adaptive_sync_client


def main():
    """ A Plato federated learning training session using Adaptive Synchronization Frequency. """
    algorithm = adaptive_sync_algorithm.Algorithm
    client = adaptive_sync_client.Client(algorithm=algorithm)
    server = fedavg.Server(algorithm=algorithm)
    server.run(client)


if __name__ == "__main__":
    main()
