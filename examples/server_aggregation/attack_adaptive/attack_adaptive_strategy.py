"""
A federated learning training session using attack-adaptive aggregation with strategy pattern.

Note: This is a simplified version. Attack-adaptive's aggregation logic is implemented
in the algorithm layer, so the server doesn't need custom strategies.

Reference:

Ching Pui Wan, Qifeng Chen, "Robust Federated Learning with Attack-Adaptive Aggregation"
Unpublished
(https://arxiv.org/pdf/2102.05257.pdf)
"""

import attack_adaptive_algorithm
import attack_adaptive_server_strategy

from plato.clients import simple


def main():
    """A Plato federated learning training session using attack-adaptive aggregation."""
    algorithm = attack_adaptive_algorithm.Algorithm
    client = simple.Client(algorithm=algorithm)
    server = attack_adaptive_server_strategy.Server(algorithm=algorithm)
    server.run(client)


if __name__ == "__main__":
    main()
