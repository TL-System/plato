"""
The implementation paper system-heterogenous federated learning through architecture search.
"""

import resnet
from sysheterofl_algorithm import Algorithm
from sysheterofl_client import Client
from sysheterofl_server import Server
from sysheterofl_trainer import ServerTrainer


def main():
    """A Plato federated learning training session using the ElasticArch algorithm."""
    model = resnet.ResnetWrapper
    server = Server(model=model, algorithm=Algorithm, trainer=ServerTrainer)
    client = Client(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
