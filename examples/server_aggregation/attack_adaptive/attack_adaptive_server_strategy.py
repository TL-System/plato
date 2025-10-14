"""
Server aggregation using attack-adaptive aggregation with strategy pattern.

Note: Attack-adaptive aggregation is implemented in the algorithm layer, not the server layer.
The original server simply delegates to algorithm.aggregate_weights(). This pattern
is already using composition correctly through the algorithm.

Reference:

Ching Pui Wan, Qifeng Chen, "Robust Federated Learning with Attack-Adaptive Aggregation"
Unpublished
(https://arxiv.org/pdf/2102.05257.pdf)
"""

from plato.servers import fedavg


class Server(fedavg.Server):
    """
    A federated learning server using the attack-adaptive algorithm.

    Note: Attack-adaptive's logic is in the algorithm layer (attack_adaptive_algorithm.py).
    This server uses the default aggregation strategy but requires the
    attack-adaptive algorithm to be passed.

    Usage:
        import attack_adaptive_algorithm
        server = Server(algorithm=attack_adaptive_algorithm.Algorithm)
    """

    # Attack-adaptive aggregation is handled by the algorithm layer
    # No custom server logic needed with strategy-based approach
    pass
