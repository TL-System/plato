"""
A federated learning training session using FedProx.

To better handle system heterogeneity, the FedProx algorithm introduced a
proximal term in the optimizer used by local training on the clients. It has
been quite widely cited and compared with in the federated learning literature.

Reference:
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
"Federated optimization in heterogeneous networks." Proceedings of Machine
Learning and Systems, 2, 429-450.

https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf
"""

import fedprox_trainer

from plato.clients import simple
from plato.servers import fedavg


def main():
    """A Plato federated learning training session using FedProx."""
    trainer = fedprox_trainer.Trainer
    client = simple.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
