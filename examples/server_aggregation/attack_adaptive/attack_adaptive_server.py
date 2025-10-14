"""
Server aggregation using attack-adaptive aggregation.

Reference:

Ching Pui Wan, Qifeng Chen, "Robust Federated Learning with Attack-Adaptive Aggregation"
Unpublished
(https://arxiv.org/pdf/2102.05257.pdf)

Comparison to FedAtt, instead of using norm distance, this algorithm uses cosine
similarity between the client and server parameters. It also applies softmax with
temperatures.
"""

from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the attack-adaptive algorithm."""

    # pylint: disable=unused-argument
    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate weight updates from the clients using attack-adaptive aggregation."""
        return await self.algorithm.aggregate_weights(
            baseline_weights, weights_received
        )
