"""
Server aggregation using FedAtt with strategy pattern.

Note: FedAtt aggregation is implemented in the algorithm layer, not the server layer.
The original server simply delegates to algorithm.aggregate_weights(). This pattern
is already using composition correctly through the algorithm.

For consistency with the strategy-based API, we can use a custom aggregation strategy
that delegates to the algorithm, or simply use the default FedAvg server with the
FedAtt algorithm.

Reference:

S. Ji, S. Pan, G. Long, X. Li, J. Jiang, Z. Huang. "Learning Private Neural Language Modeling
with Attentive Aggregation," in Proc. International Joint Conference on Neural Networks (IJCNN),
2019.

https://arxiv.org/abs/1812.07108
"""

from plato.servers import fedavg


class Server(fedavg.Server):
    """
    The federated learning server using the FedAtt algorithm.

    Note: FedAtt's logic is in the algorithm layer (fedatt_algorithm.py).
    This server uses the default aggregation strategy but requires the
    FedAtt algorithm to be passed.

    Usage:
        import fedatt_algorithm
        server = Server(algorithm=fedatt_algorithm.Algorithm)
    """

    # FedAtt aggregation is handled by the algorithm layer
    # No custom server logic needed with strategy-based approach
    pass
