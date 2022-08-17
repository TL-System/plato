import logging
from torch import nn

from plato.servers import fedavg


class fedReIdServer(fedavg.Server):
    def __init__(self, model=None, trainer=None):
        super().__init__(model, trainer)
        self.clients_belive = None
        self.total_belive = None

    def weights_received(self, weights_received):
        """Extract update directions from clients' weights."""
        self.clients_belive = [weight[1] for weight in weights_received]
        return [weight[0] for weight in weights_received]

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        self.total_belive = sum(self.clients_belive)
        if self.total_belive == 0.0:
            self.total_belive = 1.0

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        for i, update in enumerate(deltas_received):
            belive = self.clients_belive[i]
            logging.info("%d -> %f", i, belive)
            for name, delta in update.items():
                # Use weighted average by the belive of each client
                avg_update[name] += delta * (belive / self.total_belive)

        return avg_update


def main():
    """A Plato federated learning training session using a custom model."""
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
