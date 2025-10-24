"""
A self supervised learning server for Calibre to perform divergence-aware global aggregation.

After each client clusters local samples based on their encodings, there will be
local clusters where each cluster's divergence is computed as the normalized distance
between the membership encodings and the centroid. These divergence values updated from
clients guide global aggregation on the server side. Intuitively, it rejects the client
who does not have better (lower divergence) clusters. The main reason is that in the
final, each client will perform a classification, making clusters with clear boundaries
gain higher accuracy.

One interesting observation is that the representation learned by SSL presents better
performance when it is applied to imbalanced classification. Compared to other features
learned in a supervised manner, SSL features generally lead to high accuracy in all different
classes. This observation was also mentioned in the paper:
"Self-supervised Learning is More Robust to Dataset Imbalance." (accepted by NeurIPS21)
"""

import torch

from plato.servers import fedavg_personalized as personalized_server
from plato.servers.strategies.aggregation import fedavg as fedavg_strategy


class CalibreAggregationStrategy(fedavg_strategy.FedAvgAggregationStrategy):
    """Aggregation strategy that re-weights deltas using divergence scores."""

    async def aggregate_deltas(self, updates, deltas_received, context):
        server = context.server
        divergence_rates = getattr(server, "divergence_rates_received", None)

        if divergence_rates is None or len(divergence_rates) == 0:
            return await super().aggregate_deltas(updates, deltas_received, context)

        total_divergence = torch.sum(divergence_rates)
        if total_divergence.item() == 0:
            return await super().aggregate_deltas(updates, deltas_received, context)

        weighted_deltas = []
        for i, update in enumerate(deltas_received):
            rate_ratio = (divergence_rates[i] / total_divergence).item()
            weighted_update = {
                name: delta * rate_ratio for name, delta in update.items()
            }
            weighted_deltas.append(weighted_update)

        return await super().aggregate_deltas(updates, weighted_deltas, context)


class Server(personalized_server.Server):
    """A federated learning server using the Calibre algorithm."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=CalibreAggregationStrategy(),
        )
        self.divergence_rates_received = []

    def weights_received(self, weights_received):
        """Get the divergence rates from clients."""
        processed_weights = []
        divergence_values = []

        for weight_entry in weights_received:
            if isinstance(weight_entry, dict):
                if "divergence_rate" not in weight_entry:
                    raise KeyError("Missing 'divergence_rate' in client payload.")
                divergence_values.append(weight_entry["divergence_rate"])
                state_cls = type(weight_entry)
                filtered_state = state_cls(
                    (key, value)
                    for key, value in weight_entry.items()
                    if key != "divergence_rate"
                )
                processed_weights.append(filtered_state)
            elif isinstance(weight_entry, (list, tuple)) and len(weight_entry) == 2:
                processed_weights.append(weight_entry[0])
                divergence_values.append(weight_entry[1])
            else:
                raise TypeError(
                    "Unexpected weight entry format received from client: "
                    f"{type(weight_entry)!r}"
                )

        divergence_tensor = torch.stack(
            [torch.as_tensor(value) for value in divergence_values], dim=0
        )
        self.divergence_rates_received = 1 / divergence_tensor
        return processed_weights
