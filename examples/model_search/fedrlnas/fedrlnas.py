"""
Implementation of Search Phase in Federared Model Search via Reinforcement Learning (FedRLNAS).

Reference:

Yao et al., "Federated Model Search via Reinforcement Learning", in the Proceedings of ICDCS 2021.

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9546522

The Search Space of NAS is based on: Darts https://github.com/quark0/darts .
"""

import fedrlnas_client
import fedrlnas_server
from Darts.architect import Architect
from Darts.model_search import Network
from fedrlnas_algorithm import ClientAlgorithm, ServerAlgorithm

from plato.trainers.basic import Trainer


def main():
    """A Plato federated learning training session using the FedRLNAS algorithm."""
    client = fedrlnas_client.Client(
        model=Network, algorithm=ClientAlgorithm, trainer=Trainer
    )
    server = fedrlnas_server.Server(
        model=Architect, algorithm=ServerAlgorithm, trainer=Trainer
    )

    server.run(client)


if __name__ == "__main__":
    main()
