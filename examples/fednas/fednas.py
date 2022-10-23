"""
Federared Model Search via Reinforcement Learning

Reference:

Yao et al., "Federated Model Search via Reinforcement Learning", in the Proceedings of ICDCS 2021

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9546522

NAS Search Space: Darts https://github.com/quark0/darts
"""


import fednas_algorithm
import fednas_client
import fednas_server
from Darts.architect import Architect
from Darts.model_search import Network

from plato.trainers.basic import Trainer


def main():
    """A Plato federated learning training session using the FedNAS algorithm."""
    client = fednas_client.Client(
        model=Network, algorithm=fednas_algorithm.ClientAlgorithm, trainer=Trainer
    )
    server = fednas_server.Server(
        model=Architect, algorithm=fednas_algorithm.ServerAlgorithm, trainer=Trainer
    )

    server.run(client)


if __name__ == "__main__":
    main()
