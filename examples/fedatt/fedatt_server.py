"""
A federated learning server using FedAtt.

Reference:

Ji et al., "Learning Private Neural Language Modeling with Attentive Aggregation,"
in the Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN).

https://arxiv.org/abs/1812.07108
"""

from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedAtt algorithm."""

    # pylint: disable=unused-argument
    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate weight updates from the clients using FedAtt."""

        # Update server weights in a framework-specific algorithm for PyTorch only
        return await self.algorithm.aggregate_weights(
            baseline_weights, weights_received
        )
