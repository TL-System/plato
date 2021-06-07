import os
import logging
from collections import OrderedDict
from torch import nn

os.environ['config_file'] = 'examples/configs/server.yml'
from plato.servers import fedavg


class fedReIdServer(fedavg.Server):
    def __init__(self, model=None, trainer=None):
        super().__init__(model, trainer)

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract weights from the updates
        weights_received = self.extract_client_updates(updates)

        # Extract the total number of samples
        self.total_belive = sum([report.belive for (report, __) in updates])

        if self.total_belive == 0.0:
            self.total_belive = 1.0

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0].items()
        }

        for i, update in enumerate(weights_received):
            report, __ = weights_received[i]
            belive = report.belive
            logging.info("%d -> %f", i, belive)
            for name, delta in update.items():
                # Use weighted average by the number of samples
                avg_update[name] += delta * (belive / self.total_belive)

        return avg_update


def main():
    """A Plato federated learning training session using a custom model. """
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    server = fedReIdServer(model=model)
    server.run()


if __name__ == "__main__":
    main()
