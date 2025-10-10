"""
A federated learning training session using FedNova.

Reference:

Wang et al., "Tackling the Objective Inconsistency Problem in Heterogeneous Federated
Optimization", in the Proceedings of NeurIPS 2020.

https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html
"""

import fednova_client
import fednova_server


def main():
    """A Plato federated learning training session using the FedNova algorithm."""
    client = fednova_client.Client()
    server = fednova_server.Server()
    server.run(client)


if __name__ == "__main__":
    main()
