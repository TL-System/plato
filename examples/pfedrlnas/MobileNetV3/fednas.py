"""
Implement new algorithm: personalized federarted NAS.

Reference Search Space: MobileNetv3.
"""

import fednas_server
import fednas_client
import fednas_algorithm
import fednas_trainer

from model.mobilenetv3_supernet import NasDynamicModel
from model.architect import Architect


def main():
    """
    A Plato federated learning training session using PerFedRLNAS, paper unpublished.
    """
    supernet = NasDynamicModel
    client = fednas_client.Client(
        model=supernet,
        algorithm=fednas_algorithm.ClientAlgorithm,
        trainer=fednas_trainer.Trainer,
    )
    server = fednas_server.Server(
        model=Architect,
        algorithm=fednas_algorithm.ServerAlgorithm,
        trainer=fednas_trainer.Trainer,
    )

    server.run(client)


if __name__ == "__main__":
    main()
