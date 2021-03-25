#!/usr/bin/env python
"""
A Plato federated learning training session using a provided LeNet5 model.
"""

from config import Config
from servers import registry as server_registry
from clients import simple
import models


def main():
    """A Plato federated learning training session using a provided LeNet5 model. """
    __ = Config()

    model = models.lenet5.Model()
    client = simple.Client(model)
    server = server_registry.get()
    server.run(client)


if __name__ == "__main__":
    main()
