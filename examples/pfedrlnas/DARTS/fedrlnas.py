"""
Implement new algorithm: personalized federarted NAS.
"""

import fedrlnas_client
import fedrlnas_server
from fedrlnas_algorithm import ClientAlgorithm, ServerAlgorithm

from Darts.architect import Architect
from Darts.model_search import Network

from plato.trainers.basic import Trainer


def main():
    """A Plato federated learning training session using the PerFedRLNAS algorithm."""
    client = fedrlnas_client.Client(
        model=Network, algorithm=ClientAlgorithm, trainer=Trainer
    )
    server = fedrlnas_server.Server(
        model=Architect, algorithm=ServerAlgorithm, trainer=Trainer
    )

    server.run(client)


if __name__ == "__main__":
    main()
