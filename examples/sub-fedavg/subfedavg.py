"""
A federated learning training session using Sub-FedAvg.

"""

import subfedavg_server
import subfedavg_client
import subfedavg_edge
import subfedavg_trainer


def main():
    """ A Plato federated learning training session using the Sub-FedAvg algorithm. """
    trainer = subfedavg_trainer.Trainer()
    client = subfedavg_client.Client(trainer=trainer)
    server = subfedavg_server.Server()
    edge_server = subfedavg_server.Server
    edge_client = subfedavg_edge.Client
    server.run(client, edge_server, edge_client)


if __name__ == "__main__":
    main()
