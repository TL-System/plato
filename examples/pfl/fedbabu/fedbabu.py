"""
An implementation of the FedBABU algorithm.

J. Oh, et al., "FedBABU: Toward Enhanced Representation for Federated Image Classification,"
in the Proceedings of ICLR 2022.

https://openreview.net/pdf?id=HuaYQfggn5u

Source code: https://github.com/jhoon-oh/FedBABU
"""

from fedbabu_trainer import Trainer
from fedbabu_client import Client

from examples.pfl.bases.fedavg_personalized import Server


def main():
    """
    A Plato federated learning training session using the FedBABU algorithm under the
    supervised learning setting.
    """

    client = Client(trainer=Trainer)
    server = Server(trainer=Trainer)

    server.run(client)


if __name__ == "__main__":
    main()
