"""
A federated learning client using FedNova, where the local number of epochs is randomly
generated and communicated to the server at each communication round.

Reference:

Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated
Optimization", in the Proceedings of NeurIPS 2020.

https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html
"""

import logging

import numpy as np

from plato.clients import simple
from plato.config import Config


class Client(simple.Client):
    """A FedNova federated learning client who sends weight updates
    and the number of local epochs."""

    def configure(self) -> None:
        super().configure()
        np.random.seed(3000 + self.client_id)

    async def _train(self):
        """FedNova clients use different number of local epochs."""

        # Generate the number of local epochs randomly
        if (
            hasattr(Config().algorithm, "pattern")
            and Config().algorithm.pattern == "uniform_random"
        ):
            local_epochs = np.random.randint(2, Config().algorithm.max_local_epochs + 1)
            # Perform model training for a specific number of epochs
            Config().trainer = Config().trainer._replace(epochs=local_epochs)

            logging.info(
                "[Client #%d] Training with %d epochs.", self.client_id, local_epochs
            )

        # Call parent's _train method to get report and weights
        report, weights = await super()._train()

        # Add the epochs information to the report
        report.epochs = Config().trainer.epochs

        return report, weights
