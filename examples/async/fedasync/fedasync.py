"""
A federated learning training session using FedAsync.

Reference:

Xie, C., Koyejo, S., Gupta, I. "Asynchronous federated optimization,"
in Proc. 12th Annual Workshop on Optimization for Machine Learning (OPT 2020).

https://opt-ml.org/papers/2020/paper_28.pdf
"""

import fedasync_algorithm
import fedasync_server

from plato.clients import simple


def main():
    """A Plato federated learning training session using FedAsync."""
    algorithm = fedasync_algorithm.Algorithm
    client = simple.Client(algorithm=algorithm)
    server = fedasync_server.Server(algorithm=algorithm)
    server.run(client)


if __name__ == "__main__":
    main()
