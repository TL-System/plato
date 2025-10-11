"""
The implementation of FedPer method based on the plato's pFL code.

Manoj Ghuhan Arivazhagan, et al., Federated learning with personalization layers, 2019.
https://arxiv.org/abs/1912.00818

Official code: None
Third-party code: https://github.com/jhoon-oh/FedBABU
"""

import fedper_trainer

from plato.clients import fedavg_personalized as personalized_client
from plato.servers import fedavg_personalized as personalized_server


def main():
    """
    A personalized federated learning session for FedPer approach.
    """
    trainer = fedper_trainer.Trainer
    client = personalized_client.Client(trainer=trainer)
    server = personalized_server.Server(trainer=trainer)

    server.run(client)


if __name__ == "__main__":
    main()
