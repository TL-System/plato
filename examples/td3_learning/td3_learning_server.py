"""
This example uses a very simple model to show how the model and the server
be customized in Plato and executed in a standalone fashion.

To run this example:

python examples/customized/custom_server.py -c examples/customized/server.yml
"""

import logging

from torch import nn

from plato.servers import fedavg


class TD3Server(fedavg.Server):
    """ A custom federated learning server. """

    def __init__(self, model=None, trainer=None, algorithm=None):
        super().__init__(trainer=trainer, algorithm=algorithm,model=model)
        logging.info("A custom server has been initialized.")

