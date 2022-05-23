import os
import logging
from torch import nn

os.environ['config_file'] = 'examples/configs/server.yml'
from plato.servers import fedavg


class fedReIdServer(fedavg.Server):

    def __init__(self, model=None, trainer=None):
        super().__init__(model, trainer)
        self.clients_belive = None

    def compute_weight_deltas(self, updates):
        """ Extract the model weights and update directions from clients updates. """
        weights_received = [payload[0] for (__, payload, __) in updates]

        self.clients_belive = [payload[1] for (__, payload, __) in updates]

        return self.algorithm.compute_weight_deltas(weights_received)

    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""
        # Extract weights from the updates
        deltas_received = self.compute_weight_deltas(updates)

        self.total_belive = sum(self.clients_belive)
        if self.total_belive == 0.0:
            self.total_belive = 1.0

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            report, __ = deltas_received[i]
            belive = self.clients_belive[i]
            logging.info("%d -> %f", i, belive)
            for name, delta in update.items():
                # Use weighted average by the belive of each client
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
