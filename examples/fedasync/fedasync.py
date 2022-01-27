"""
A federated learning training session using FedAsync.

Reference:

Xie, C., Koyejo, S., Gupta, I. (2019). "Asynchronous federated optimization,"
in Proc. 12th Annual Workshop on Optimization for Machine Learning (OPT 2020).

https://opt-ml.org/papers/2020/paper_28.pdf
"""
import os

os.environ['config_file'] = './fedasync_MNIST_lenet5.yml'

from plato.trainers import basic
from plato.clients import simple

import fedasync_server


def main():
    """ A Plato federated learning training session using FedAsync. """
    trainer = basic.Trainer()
    client = simple.Client(trainer=trainer)
    server = fedasync_server.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
