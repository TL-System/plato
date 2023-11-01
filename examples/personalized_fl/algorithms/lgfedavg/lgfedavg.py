"""
The implementation of LG-FedAvg method based on the plato's pFL code.

Paul Pu Liang, et al., Think Locally, Act Globally: Federated Learning
with Local and Global Representations. https://arxiv.org/abs/2001.01523

"""

from pflbases import fedavg_partial
from pflbases import fedavg_personalized
from plato.clients.simple import Client
import lgfedavg_trainer


def main():
    """
    A Plato personalized federated learning session for LG-FedAvg approach.
    """
    trainer = lgfedavg_trainer.Trainer
    client = Client(
        trainer=trainer,
        algorithm=fedavg_partial.Algorithm,
    )
    server = fedavg_personalized.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
