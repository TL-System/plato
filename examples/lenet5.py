#!/usr/bin/env python
"""
A Plato federated learning training session using a provided config file and the LeNet5 model.
"""
import os
os.environ['config_file'] = 'configs/MNIST/fedavg_lenet5.yml'

from models import lenet5
from clients import simple
from servers import registry as server_registry


def main():
    """A Plato federated learning training session using a provided LeNet5 model. """
    model = lenet5.Model()
    client = simple.Client(model)
    server = server_registry.get()
    server.run(client)


if __name__ == "__main__":
    main()
