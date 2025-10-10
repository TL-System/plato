"""
Server aggregation using FedAtt.

Reference:

S. Ji, S. Pan, G. Long, X. Li, J. Jiang, Z. Huang. "Learning Private Neural Language Modeling
with Attentive Aggregation," in Proc. International Joint Conference on Neural Networks (IJCNN),
2019.

https://arxiv.org/abs/1812.07108
"""

from plato.servers import fedavg


class Server(fedavg.Server):
    """The federated learning server using the FedAtt algorithm."""

    # pylint: disable=unused-argument
    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate weight updates from the clients using FedAtt."""
        return await self.algorithm.aggregate_weights(
            baseline_weights, weights_received
        )
