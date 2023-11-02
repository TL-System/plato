"""
The implementation of LG-FedAvg method based on the plato's pFL code.

Paul Pu Liang, et al., Think Locally, Act Globally: Federated Learning
with Local and Global Representations. https://arxiv.org/abs/2001.01523

"""

import lgfedavg_trainer

from plato.servers import fedavg_personalized as personalized_server
from plato.clients import fedavg_personalized as personalized_client


def main():
    """
    A Plato personalized federated learning session for LG-FedAvg approach.
    """
    trainer = lgfedavg_trainer.Trainer
    client = personalized_client.Client(trainer=trainer)
    server = personalized_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
