"""
An implementation of LG-FedAvg.

P. Liang, et al., "Think Locally, Act Globally: Federated Learning
with Local and Global Representations," Arxiv 2020.

https://arxiv.org/abs/2001.01523

Source code: https://github.com/pliang279/LG-FedAvg
"""

import lgfedavg_trainer

from plato.clients import fedavg_personalized as personalized_client
from plato.servers import fedavg_personalized as personalized_server


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
