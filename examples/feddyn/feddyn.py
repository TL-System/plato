"""
An implementation of the FedDyn algorithm.

D. Acar, et al., "Federated Learning Based on Dynamic Regularization," in the
Proceedings of ICLR 2021.

https://openreview.net/forum?id=B7v4QMR6Z9w

Source code: https://github.com/alpemreacar/FedDyn
"""
from plato.servers import fedavg
from plato.clients import simple

import feddyn_trainer


def main():
    """A Plato federated learning training session using FedDyn."""
    trainer = feddyn_trainer.Trainer
    client = simple.Client(trainer=trainer)
    server = fedavg.Server()
    server.run(client)


if __name__ == "__main__":
    main()
