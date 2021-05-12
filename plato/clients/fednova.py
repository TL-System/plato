"""
A federated learning client whose local number of epochs is randomly
generated and communicated to the server at each communication round.
"""

import logging
import random
from dataclasses import dataclass

from plato.config import Config

from plato.clients import simple


@dataclass
class Report(simple.Report):
    """A client report containing the number of local epochs."""
    epochs: int


class Client(simple.Client):
    """A fednova federated learning client who sends weight updates
    and the number of local epochs."""
    def __init__(self):
        super().__init__()
        self.pattern = None
        self.max_local_iter = None
        random.seed(3000 + int(self.client_id))

    async def train(self):
        """FedNova clients use different number of local epochs."""

        # generate the number of local epochs randomly
        if Config().algorithm.pattern == "constant":
            local_epochs = Config().algorithm.max_local_epochs
        else:
            local_epochs = random.randint(2,
                                          Config().algorithm.max_local_epochs)

        logging.info("[Client #%s] Training with %d epoches.", self.client_id,
                     local_epochs)

        # Perform model training for a specific number of epoches
        Config().trainer = Config().trainer._replace(epochs=local_epochs)

        return await super().train()
