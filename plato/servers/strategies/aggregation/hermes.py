"""
Hermes aggregation strategy.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch

from plato.servers.strategies.aggregation.fedavg import FedAvgAggregationStrategy
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class HermesAggregationStrategy(AggregationStrategy):
    """
    Aggregation strategy for the Hermes personalization algorithm.

    Hermes selectively averages overlapping parameters using the pruning masks
    reported by each client, while falling back to standard FedAvg semantics
    for unmasked layers.
    """

    def __init__(self):
        self._fedavg = FedAvgAggregationStrategy()

    def setup(self, context: ServerContext) -> None:
        self._fedavg.setup(context)

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        return await self._fedavg.aggregate_deltas(updates, deltas_received, context)

    async def aggregate_weights(
        self,
        updates: List[SimpleNamespace],
        baseline_weights: Dict,
        weights_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        server = context.server
        trainer = context.trainer
        algorithm = context.algorithm

        total_samples = sum(update.report.num_samples for update in updates)
        server.total_samples = total_samples

        masks_received = getattr(server, "masks_received", None)
        if not masks_received:
            return None

        weights_numpy: List[Dict[str, np.ndarray]] = []
        for weight_dict in weights_received:
            weights_numpy.append(
                {
                    name: tensor.detach().cpu().numpy().copy()
                    for name, tensor in weight_dict.items()
                }
            )

        masked_layers = []
        for name, layer in trainer.model.named_parameters():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                masked_layers.append(f"{name}.weight")

        step = 0
        num_clients = len(weights_numpy)

        for layer_name in weights_numpy[0].keys():
            if layer_name in masked_layers:
                mask_count = np.zeros_like(masks_received[0][step].reshape([-1]))
                avg = np.zeros_like(weights_numpy[0][layer_name].reshape([-1]))

                for index in range(num_clients):
                    num_samples = updates[index].report.num_samples
                    mask = masks_received[index][step].reshape([-1])
                    mask_count += mask
                    avg += (
                        weights_numpy[index][layer_name].reshape([-1])
                        * num_samples
                        / total_samples
                    )

                mask_count = np.where(mask_count == num_clients, 1, 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    final_avg = np.divide(
                        avg, mask_count, out=np.zeros_like(avg), where=mask_count != 0
                    )

                valid_indices = np.isfinite(final_avg)
                for index in range(num_clients):
                    flattened = weights_numpy[index][layer_name].reshape([-1])
                    flattened[valid_indices] = final_avg[valid_indices]
                    weights_numpy[index][layer_name] = flattened.reshape(
                        weights_numpy[index][layer_name].shape
                    )

                step += 1
            else:
                avg = np.zeros_like(
                    weights_numpy[0][layer_name].reshape([-1]), dtype=np.float64
                )

                for index in range(num_clients):
                    num_samples = updates[index].report.num_samples
                    avg += weights_numpy[index][layer_name].reshape([-1]) * (
                        num_samples / total_samples
                    )

                reshaped = avg.reshape(weights_numpy[0][layer_name].shape)
                for index in range(num_clients):
                    weights_numpy[index][layer_name] = reshaped.copy()

        aggregated_weights: List[Dict[str, torch.Tensor]] = []
        for weight_dict in weights_numpy:
            aggregated_weights.append(
                {
                    name: torch.from_numpy(array).to(
                        dtype=baseline_weights[name].dtype,
                        device=baseline_weights[name].device,
                    )
                    for name, array in weight_dict.items()
                }
            )

        server.update_client_model(aggregated_weights, updates)

        deltas_received = algorithm.compute_weight_deltas(
            baseline_weights, aggregated_weights
        )

        avg_deltas = await self._fedavg.aggregate_deltas(
            updates, deltas_received, context
        )

        updated_weights = algorithm.update_weights(avg_deltas)
        return updated_weights
