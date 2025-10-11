"""
Personalized cross-silo federated learning using the MAML algorithm
"""

import sys

sys.path.append("../fl_maml/")

# pylint: disable=import-error
# pylint: disable=wrong-import-position
import cs_maml_edge
import cs_maml_server
import fl_maml_client as cs_maml_client
import fl_maml_trainer as cs_maml_trainer


def main():
    """A Plato federated learning training session using the MAML algorithm."""
    trainer = cs_maml_trainer.Trainer
    client = cs_maml_client.Client(trainer=trainer())
    server = cs_maml_server.Server(trainer=trainer())
    edge_server = cs_maml_server.Server
    edge_client = cs_maml_edge.Client
    server.run(client, edge_server, edge_client, trainer)


if __name__ == "__main__":
    main()
