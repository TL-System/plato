from collections import OrderedDict

from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """The federated learning algorithm for FedAsync, used by the server."""

    async def aggregate_weights(
        self, baseline_weights, weights_received, mixing=0.9, **kwargs
    ):
        """Aggregates the weights received into baseline weights."""
        # Actually update the global model's weights
        updated_weights = OrderedDict()

        for name, weight in baseline_weights.items():
            updated_weights[name] = (
                weight * (1 - mixing) + weights_received[0][name] * mixing
            )

        return updated_weights
