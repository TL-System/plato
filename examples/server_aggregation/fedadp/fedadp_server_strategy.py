"""
FedADP aggregation strategy using adaptive weighting based on gradient angles.

Reference:

H. Wu, P. Wang. "Fast-Convergent Federated Learning with Adaptive Weighting," in IEEE Trans.
on Cognitive Communications and Networking (TCCN), 2021.

https://ieeexplore.ieee.org/abstract/document/9442814
"""

import math
from types import SimpleNamespace
from typing import Dict, List

import numpy as np

from plato.config import Config
from plato.servers.strategies.base import AggregationStrategy, ServerContext


class FedADPAggregationStrategy(AggregationStrategy):
    """FedADP aggregation with adaptive weighting driven by gradient angles."""

    def __init__(self, alpha: float = 5):
        super().__init__()
        self.alpha = alpha
        self.local_angles: Dict[int, float] = {}
        self.last_global_grads = None
        self.global_grads = None
        self.adaptive_weighting = None

    def setup(self, context: ServerContext) -> None:
        """Initialize alpha from the configuration if provided."""
        if hasattr(Config().algorithm, "alpha"):
            self.alpha = Config().algorithm.alpha

    async def aggregate_deltas(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ) -> Dict:
        """Aggregate client updates using the FedADP adaptive weighting scheme."""
        num_samples = [update.report.num_samples for update in updates]
        total_samples = sum(num_samples)

        self.global_grads = {
            name: context.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        for idx, delta in enumerate(deltas_received):
            weight = num_samples[idx] / total_samples if total_samples > 0 else 0.0
            for name, value in delta.items():
                self.global_grads[name] += value * weight

        # Preserve the current global gradients dictionary before flattening
        self.last_global_grads = dict(self.global_grads)

        self.adaptive_weighting = self.calc_adaptive_weighting(
            updates, deltas_received, num_samples, context
        )

        avg_update = {
            name: context.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        for idx, delta in enumerate(deltas_received):
            weight = self.adaptive_weighting[idx]
            for name, value in delta.items():
                avg_update[name] += value * weight

        return avg_update

    def calc_adaptive_weighting(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        num_samples: List[int],
        context: ServerContext,
    ):
        """Compute aggregation weights considering node contribution and data size."""
        contribs = self.calc_contribution(updates, deltas_received, context)

        adaptive_weighting = [0.0] * len(deltas_received)
        total_weight = 0.0
        for idx, contrib in enumerate(contribs):
            total_weight += num_samples[idx] * math.exp(contrib)

        if total_weight == 0.0:
            # Fallback to uniform weighting if total weight is zero.
            uniform_weight = 1.0 / len(deltas_received) if deltas_received else 0.0
            return [uniform_weight for _ in deltas_received]

        for idx, contrib in enumerate(contribs):
            adaptive_weighting[idx] = (
                num_samples[idx] * math.exp(contrib)
            ) / total_weight

        return adaptive_weighting

    def calc_contribution(
        self,
        updates: List[SimpleNamespace],
        deltas_received: List[Dict],
        context: ServerContext,
    ):
        """Calculate node contribution based on the angle between local and global gradients."""
        num_clients = len(deltas_received)
        angles = [0.0] * num_clients
        contribs = [0.0] * num_clients

        global_grads = self.process_grad(self.global_grads)
        self.global_grads = global_grads

        for idx, update in enumerate(deltas_received):
            local_grads = self.process_grad(update)
            inner = np.inner(global_grads, local_grads)
            norms = np.linalg.norm(global_grads) * np.linalg.norm(local_grads)
            if norms == 0:
                angles[idx] = 0.0
            else:
                angles[idx] = np.arccos(np.clip(inner / norms, -1.0, 1.0))

        current_round = context.current_round if context.current_round > 0 else 1

        for idx, angle in enumerate(angles):
            client_id = updates[idx].client_id

            if client_id not in self.local_angles:
                self.local_angles[client_id] = angle

            self.local_angles[client_id] = (
                (current_round - 1) / current_round
            ) * self.local_angles[client_id] + (1 / current_round) * angle

            alpha = (
                Config().algorithm.alpha
                if hasattr(Config().algorithm, "alpha")
                else self.alpha
            )

            contribs[idx] = alpha * (
                1 - math.exp(-math.exp(-alpha * (self.local_angles[client_id] - 1)))
            )

        return contribs

    @staticmethod
    def process_grad(grads):
        """Convert gradients to a flattened 1-D array."""
        grads = list(dict(sorted(grads.items(), key=lambda x: x[0].lower())).values())

        flattened = grads[0]
        for idx in range(1, len(grads)):
            flattened = np.append(
                flattened, -grads[idx] / Config().parameters.optimizer.lr
            )

        return flattened
