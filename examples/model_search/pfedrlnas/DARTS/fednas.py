"""
Implement new algorithm: personalized federarted NAS.
"""

import fednas_client
import fednas_server
from Darts.architect import Architect
from Darts.model_search import Network
from fednas_algorithm import ClientAlgorithm, ServerAlgorithm

from plato.trainers.basic import Trainer


def main():
    """A Plato federated learning training session using the PerFedRLNAS algorithm."""
    client = fednas_client.Client(
        model=Network, algorithm=ClientAlgorithm, trainer=Trainer
    )
    server = fednas_server.Server(
        model=Architect, algorithm=ServerAlgorithm, trainer=Trainer
    )

    server.run(client)


if __name__ == "__main__":
    main()
