"""
An Plato implementation of system-heterogenous federated learning through architecture search.

Reference: D. Yao, "Exploring System-Heterogeneous Federated Learning with Dynamic Model Selection,"
https://arxiv.org/abs/2409.08858.
"""

import resnet
from sysheterofl_algorithm import Algorithm
from sysheterofl_client import create_client
from sysheterofl_server import Server
from sysheterofl_trainer import ServerTrainer


def main():
    """A Plato federated learning training session."""
    model = resnet.ResnetWrapper
    server = Server(model=model, algorithm=Algorithm, trainer=ServerTrainer)
    client = create_client(model=model)

    server.run(client)


if __name__ == "__main__":
    main()
