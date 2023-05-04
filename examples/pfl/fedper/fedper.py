"""
The implementation of FedPer method based on the plato's pFL code.

Manoj Ghuhan Arivazhagan, et.al, Federated learning with personalization layers, 2019.
https://arxiv.org/abs/1912.00818

Official code: None
Third-part code: https://github.com/jhoon-oh/FedBABU
"""

import fedper_client
import fedper_trainer

from examples.pfl.bases.fedavg_personalized import Server


def main():
    """
    A Plato federated learning training session using the FedBABU algorithm under the
    supervised learning setting.
    """
    trainer = fedper_trainer.Trainer
    client = fedper_client.Client(trainer=trainer)
    server = Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
