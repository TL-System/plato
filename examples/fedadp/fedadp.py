"""
A federated learning training session using the FedAdp algorithm.

Reference:

Wu et al., "Fast-Convergent Federated Learning with Adaptive Weighting,"
in IEEE Transactions on Cognitive Communications and Networking (TCCN'21).

https://ieeexplore.ieee.org/abstract/document/9442814
"""
import os


os.environ['config_file'] = './fedadp_MNIST_lenet5.yml'

import fedadp_server


def main():
    """ A Plato federated learning training session using the FedAdp algorithm. """
    server = fedadp_server.Server()
    server.run()


if __name__ == "__main__":
    main()
