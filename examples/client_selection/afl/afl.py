"""
A federated learning server using Active Federated Learning, where in each round
clients are selected not uniformly at random, but with a probability conditioned
on the current model, as well as the data on the client, to maximize efficiency.

Reference:

Goetz et al., "Active Federated Learning", 2019.

https://arxiv.org/pdf/1909.12641.pdf
"""

import afl_client
import afl_server


def main():
    """A Plato federated learning training session using the AFL algorithm."""
    client = afl_client.Client()
    server = afl_server.Server()
    server.run(client)


if __name__ == "__main__":
    main()
