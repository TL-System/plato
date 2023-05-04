"""
The implementation of LG-FedAvg method based on the plato's
pFL code.

Paul Pu Liang, et.al, Think Locally, Act Globally: Federated Learning with Local and Global Representations
https://arxiv.org/abs/2001.01523

Official code: https://github.com/pliang279/LG-FedAvg

"""

import lgfedavg_trainer
import lgfedavg_client

from examples.pfl.bases import fedavg_personalized


def main():
    """An interface for running the LG-FedAvg method."""

    trainer = lgfedavg_trainer.Trainer
    client = lgfedavg_client.Client(trainer=trainer)
    server = fedavg_personalized.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
