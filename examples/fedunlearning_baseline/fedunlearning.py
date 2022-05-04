"""
A federated unlearning model to enables data holders to proactively erase their data from a trained model.

Reference: Liu et al., "The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining." in Proc. INFOCOM, 2022 https://arxiv.org/abs/2203.07320

"""

import os

os.environ[
    'config_file'] = 'examples/fedunlearning_baseline/fedun_MNIST_lenet5.yml'

import fedunlearning_client
from plato.servers import fedavg


def main():
    """ A Plato federated learning training session using the FedSarah algorithm. """
    client = fedunlearning_client.Client()
    server = fedavg.Server()
    server.run(client)


if __name__ == "__main__":
    main()