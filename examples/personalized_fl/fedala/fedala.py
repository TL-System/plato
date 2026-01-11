"""
A federated learning training session using FedALA.

Reference:
Zhang, C., Xie, X., Tian, H., Wang, J., & Xu, Y. (2022).
"FedALA: Adaptive Local Aggregation for Federated Learning."
https://arxiv.org/abs/2212.01197
"""

import fedala_trainer

from plato.clients import simple
from plato.servers import fedavg


def main():
    """A Plato federated learning training session using FedALA."""
    trainer = fedala_trainer.Trainer
    client = simple.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
