"""
A federated learning training session using FedAtt.

Reference:

S. Ji, S. Pan, G. Long, X. Li, J. Jiang, Z. Huang. "Learning Private Neural Language Modeling
with Attentive Aggregation," in Proc. International Joint Conference on Neural Networks (IJCNN),
2019.

https://arxiv.org/abs/1812.07108
"""

import fedatt_algorithm
import fedatt_server


def main():
    """A Plato federated learning training session using the FedAtt algorithm."""
    algorithm = fedatt_algorithm.Algorithm
    server = fedatt_server.Server(algorithm=algorithm)
    server.run()


if __name__ == "__main__":
    main()
