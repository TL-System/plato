"""
A federated learning training session using Active Federated Learning with strategy pattern.

This is the updated version using the strategy-based API instead of inheritance.

Reference:

Goetz et al., "Active Federated Learning", 2019.

https://arxiv.org/pdf/1909.12641.pdf
"""

import afl_client
import afl_server_strategy


def main():
    """A Plato federated learning training session using AFL strategy."""
    client = afl_client.Client()
    server = afl_server_strategy.Server()
    server.run(client)


if __name__ == "__main__":
    main()
