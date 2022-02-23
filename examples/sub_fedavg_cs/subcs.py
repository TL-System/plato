"""
A federated learning training session using Sub-FedAvg(Un)
in three-layer cross-silo federated learning.

"""
import sys

sys.path.append("../sub_fedavg/")

# pylint: disable=import-error
# pylint: disable=wrong-import-position
import subfedavg_trainer as subcs_trainer
import subfedavg_client as subcs_client
import subcs_server
import subcs_edge


def main():
    """ A Plato federated learning training session using the Sub-FedAvg(Un) algorithm. """
    trainer = subcs_trainer.Trainer()
    client = subcs_client.Client(trainer=trainer)
    server = subcs_server.Server()
    edge_server = subcs_server.Server
    edge_client = subcs_edge.Client
    server.run(client, edge_server, edge_client)


if __name__ == "__main__":
    main()
