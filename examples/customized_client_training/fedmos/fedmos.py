"""
An implementation of the FedMos algorithm.

X. Wang, Y. Chen, Y. Li, X. Liao, H. Jin and B. Li, "FedMoS: Taming Client Drift in Federated Learning with Double Momentum and Adaptive Selection," IEEE INFOCOM 2023

Paper: https://ieeexplore.ieee.org/document/10228957

Source code: https://github.com/Distributed-Learning-Networking-Group/FedMoS
"""

import fedmos_trainer

from plato.clients import simple
from plato.servers import fedavg


def main():
    """A Plato federated learning training session using FedDyn."""
    trainer = fedmos_trainer.Trainer
    client = simple.Client(trainer=trainer)
    server = fedavg.Server()
    server.run(client)


if __name__ == "__main__":
    main()
