"""
An implementation of the FedDyn algorithm.

D. Acar, et al., "Federated Learning Based on Dynamic Regularization," in the
Proceedings of ICLR 2021.

https://openreview.net/forum?id=B7v4QMR6Z9w

Source code: https://github.com/alpemreacar/FedDyn
"""

import feddyn_client
import feddyn_server


def main():
    """A Plato federated learning training session using FedDyn."""
    client = feddyn_client.create_client()
    server = feddyn_server.Server()
    server.run(client)


if __name__ == "__main__":
    main()
