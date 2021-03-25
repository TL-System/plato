#!/usr/bin/env python
"""
A Plato federated learning training session using a provided LeNet5 model.
"""

from config import Config
from servers import registry as server_registry
import clients
import models


def main():
    """A Plato federated learning training session using a provided LeNet5 model. """
    __ = Config()

    model = models.lenet5.Model()
    client = clients.SimpleClient(model)
    server = server_registry.get()
    server.run(client)


if __name__ == "__main__":
    main()
