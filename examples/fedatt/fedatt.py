"""
A federated learning training session using FedAtt.

Reference:

Ji et al., "Learning Private Neural Language Modeling with Attentive Aggregation,"
in the Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN).

https://arxiv.org/pdf/1812.07108.pdf
"""
import os

os.environ['config_file'] = 'examples/fedatt/fedatt_MNIST_lenet5.yml'

import fedatt_server


def main():
    """ A Plato federated learning training session using the fedatt algorithm. """
    server = fedatt_server.Server()
    server.run()


if __name__ == "__main__":
    main()
