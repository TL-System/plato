"""
A federated learning training session using FedAtt with strategy pattern.

Note: This is a simplified version. FedAtt's aggregation logic is implemented
in the algorithm layer, so the server doesn't need custom strategies.

Reference:

S. Ji, S. Pan, G. Long, X. Li, J. Jiang, Z. Huang. "Learning Private Neural Language Modeling
with Attentive Aggregation," in Proc. International Joint Conference on Neural Networks (IJCNN),
2019.

https://arxiv.org/abs/1812.07108
"""

import fedatt_algorithm
import fedatt_server_strategy

from plato.clients import simple


def main():
    """A Plato federated learning training session using FedAtt."""
    algorithm = fedatt_algorithm.Algorithm
    client = simple.Client(algorithm=algorithm)
    server = fedatt_server_strategy.Server(algorithm=algorithm)
    server.run(client)


if __name__ == "__main__":
    main()
