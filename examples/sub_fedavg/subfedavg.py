"""
A federated learning training session using Sub-FedAvg(Un).

S. Vahidian, M. Morafah, and B. Lin,
“Personalized Federated Learning by Structured and Unstructured Pruning Under Data Heterogeneity,”
in 41st IEEE International Conference on Distributed Computing Systems Workshops (ICDCSW). 2021.

Original sourcecode: https://github.com/MMorafah/Sub-FedAvg

"""

import subfedavg_client
import subfedavg_trainer

from plato.servers import fedavg


def main():
    """A Plato federated learning training session using the Sub-FedAvg algorithm."""
    trainer = subfedavg_trainer.Trainer
    client = subfedavg_client.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
