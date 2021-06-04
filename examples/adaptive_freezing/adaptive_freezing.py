"""
A federated learning client with support for Adaptive Parameter Freezing (APF).

Reference:

C. Chen, H. Xu, W. Wang, B. Li, B. Li, L. Chen, G. Zhang. “Communication-
Efficient Federated Learning with Adaptive Parameter Freezing,” in the
Proceedings of the 41st IEEE International Conference on Distributed Computing
Systems (ICDCS 2021), Online, July 7-10, 2021 (found in docs/papers).
"""
import os

os.environ['config_file'] = 'adaptive_freezing_MNIST_lenet5.yml'

from plato.servers import fedavg
from plato.trainers import basic

import adaptive_freezing_client
import adaptive_freezing_algorithm

def main():
    """ A Plato federated learning training session using the FedNova algorithm. """
    trainer = basic.Trainer()
    algorithm = adaptive_freezing_algorithm.Algorithm(trainer=trainer)
    client = adaptive_freezing_client.Client(algorithm=algorithm, trainer=trainer)
    server = fedavg.Server(algorithm=algorithm, trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()
