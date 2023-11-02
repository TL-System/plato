"""
The implementation of LG-FedAvg method based on the plato's pFL code.

Paul Pu Liang, et al., Think Locally, Act Globally: Federated Learning
with Local and Global Representations. https://arxiv.org/abs/2001.01523

"""

from pflbases import fedavg_personalized
from pflbases import fedavg_personalized
from pflbases import fedavg_personalized
import lgfedavg_trainer


def main():
    """
    A Plato personalized federated learning session for LG-FedAvg approach.
    """
    trainer = lgfedavg_trainer.Trainer
    client = fedavg_personalized.Client(
        trainer=trainer,
        algorithm=fedavg_personalized.Algorithm,
    )
    server = fedavg_personalized.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
