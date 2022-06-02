"""
This example uses a very simple model to show how the model and the server
be customized in Plato and executed in a standalone fashion.

To run this example:

python examples/customized/custom_server.py -c examples/customized/server.yml
"""

import logging

from torch import nn

from plato.servers import fedavg


class CustomServer(fedavg.Server):
    """ A custom federated learning server. """

    def __init__(self, model=None, trainer=None):
        super().__init__(model, trainer)
        logging.info("A custom server has been initialized.")


def main():
    """ A Plato federated learning training session using a custom model. """
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    server = CustomServer(model=model)
    server.run()


if __name__ == "__main__":
    main()
