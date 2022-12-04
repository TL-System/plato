"""
The implementation of FedBABU method.

Jaehoon Oh, et.al, FedBABU: Toward Enhanced Representation for Federated Image Classification.
in the Proceedings of ICML 2021.

Paper address: https://openreview.net/pdf?id=HuaYQfggn5u
Official code: https://github.com/jhoon-oh/FedBABU

"""

from fedbabu_trainer import Trainer

from plato.servers.fedavg import Server
from plato.clients.simple import Client


def main():
    """An interface for running the FedBABU method under the
    supervised learning setting.
    """

    client = Client(trainer=Trainer)
    server = Server(trainer=Trainer)

    server.run(client)


if __name__ == "__main__":
    main()
