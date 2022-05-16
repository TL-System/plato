"""
A federated learning training session using FedProx.

Reference:
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
"Federated optimization in heterogeneous networks." Proceedings of Machine
Learning and Systems, 2, 429-450.

https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf
"""
import fedprox_trainer

from plato.servers import fedavg
from plato.clients import simple


def main():
    """ A Plato federated learning training session using FedProx. """
    trainer = fedprox_trainer.Trainer
    client = simple.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)


if __name__ == "__main__":
    main()
