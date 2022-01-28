"""
A federated learning server using Port.

Reference:

"How Asynchronous can Federated Learning Be?"

"""
import os

os.environ['config_file'] = 'examples/port_fedasync_mode/port_MNIST_lenet5.yml'

from plato.trainers import basic
from plato.clients import simple

import port_server


def main():
    """ A Plato federated learning training session using FedAsync. """
    trainer = basic.Trainer()
    client = simple.Client(trainer=trainer)
    server = port_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
