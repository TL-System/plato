"""
A FedAvg server using norm bounding defense

Reference:
Sun, Z., Kairouz, P., Suresh, A. T., & McMahan, H. B. (2019). Can you really backdoor
federated learning?. arXiv preprint arXiv:1911.07963.

https://arxiv.org/pdf/1911.07963.pdf
"""

import asyncio

import numpy as np

from plato.config import Config
from plato.servers import fedavg


def compute_weights_norm(model_weight_dict):
    """Compute the norm of the model update from a client."""
    weights_norm = 0
    for weight in model_weight_dict.values():
        weights_norm += np.sum(np.array(weight) ** 2)
    weights_norm = np.sqrt(weights_norm)
    return weights_norm


class Server(fedavg.Server):
    """FedAvg server with norm bounding defense"""

    async def aggregate_deltas(self, updates, deltas_received):
        """Aggregate weight updates from the clients using federated averaging."""
        norm_bound = (
            Config().server.norm_bounding_threshold
            if hasattr(Config().server, "norm_bounding_threshold")
            else None
        )

        # Perform weighted averaging
        avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in deltas_received[0].items()
        }

        for _, update in enumerate(deltas_received):
            weight_norm = compute_weights_norm(update)

            for name, delta in update.items():
                # Scale client updates with norm bounding
                avg_update[name] += delta / max(1, weight_norm / norm_bound)

            # Yield to other tasks in the server
            await asyncio.sleep(0)

        return avg_update
