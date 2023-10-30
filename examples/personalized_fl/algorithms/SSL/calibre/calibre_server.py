"""
A federated learning server using Calibre,
thus is divergence-aware.
"""

import torch

from pflbases import fedavg_personalized_server


class Server(fedavg_personalized_server.Server):
    """A federated learning server using the Hermes algorithm."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.divergence_rates_received = []

    async def aggregate_deltas(self, updates, deltas_received):
        """Add the divergence rates to deltas."""

        total_divergence = torch.sum(self.divergence_rates_received)
        # normalize the delta with the divergence rates
        for i, update in enumerate(deltas_received):
            divergence_rate = self.divergence_rates_received[i]
            for name, delta in update.items():
                update[name] = delta * (divergence_rate / total_divergence)

            deltas_received[i] = update

        return await super().aggregate_deltas(updates, deltas_received)

    def weights_received(self, weights_received):
        """Get the divergenec rates sent from clients."""
        self.divergence_rates_received = torch.stack(
            [weight[1] for weight in weights_received], dim=0
        )
        print("self.divergence_rates_received: ", self.divergence_rates_received)
        return [weight[0] for weight in weights_received]
