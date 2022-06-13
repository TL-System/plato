"""
A federated learning client with support for Adaptive Parameter Freezing (APF).

Reference:

C. Chen, H. Xu, W. Wang, B. Li, B. Li, L. Chen, G. Zhang. “Communication-
Efficient Federated Learning with Adaptive Parameter Freezing,” in the
Proceedings of the 41st IEEE International Conference on Distributed Computing
Systems (ICDCS 2021), Online, July 7-10, 2021.

The camera-ready manuscript of this paper is located at:
https://ieeexplore.ieee.org/document/9546506

"""

from plato.servers import fedavg

import adaptive_freezing_client
import adaptive_freezing_algorithm


def main():
    """ A Plato federated learning training session using Adaptive Parameter Freezing. """
    algorithm = adaptive_freezing_algorithm.Algorithm
    client = adaptive_freezing_client.Client(algorithm=algorithm)
    server = fedavg.Server(algorithm=algorithm)
    server.run(client)


if __name__ == "__main__":
    main()
