"""
A federated learning training session using Sub-FedAvg(Un)
in three-layer cross-silo federated learning.

"""

import subfedavg_client as subcs_client
import subfedavg_trainer as subcs_trainer

from plato.clients import edge
from plato.servers import fedavg_cs


def main():
    """A Plato federated learning training session using the Sub-FedAvg(Un) algorithm."""
    trainer = subcs_trainer.Trainer
    client = subcs_client.Client(trainer=trainer)
    server = fedavg_cs.Server()
    edge_server = fedavg_cs.Server
    edge_client = edge.Client
    server.run(client, edge_server, edge_client)


if __name__ == "__main__":
    main()
