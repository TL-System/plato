"""
implement new algorithm: Personalized Federated NAS

Search Space: DARTS https://github.com/quark0/darts
"""

import fednas_client
import fednas_server
from plato.trainers.basic import Trainer
import fednas_algorithm
from Darts.architect import Architect
from Darts.model_search import Custom

import torch

torch.multiprocessing.set_sharing_strategy("file_system")


def main():
    client = fednas_client.Client(
        model=Custom, algorithm=fednas_algorithm.ClientAlgorithm, trainer=Trainer
    )
    server = fednas_server.Server(
        model=Architect, algorithm=fednas_algorithm.ServerAlgorithm, trainer=Trainer
    )

    server.run(client)


if __name__ == "__main__":
    main()
