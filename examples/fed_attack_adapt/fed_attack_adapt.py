"""
A federated learning training session using fed_attack_adapt.

Reference:

Ching Pui Wan, Qifeng Chen, "Robust Federated Learning with Attack-Adaptive Aggregation"
unpublished
(https://arxiv.org/pdf/2102.05257.pdf)

"""
import os

os.environ['config_file'] = 'fed_attack_adapt_MNIST_lenet5.yml'

import fed_attack_adapt_server


def main():
    """ A Plato federated learning training session using the fedatt algorithm. """
    server = fed_attack_adapt_server.Server()
    server.run()


if __name__ == "__main__":
    main()
