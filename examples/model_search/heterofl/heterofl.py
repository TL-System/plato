"""
A federated learning training session using HeteroFL

Enmao Diao, Jie Ding, and Vahid Tarokh
“HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients,”
in ICLR, 2021.

Reference "https://github.com/dem123456789/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients".
"""

import resnet
from heterofl_algorithm import Algorithm
from heterofl_client import Client
from heterofl_server import Server
from heterofl_trainer import ServerTrainer
from mobilenetv3 import MobileNetV3

from plato.config import Config


def main():
    """A Plato federated learning training session using the HeteroFL algorithm."""
    if "mobilenet" in Config().trainer.model_name:
        model = MobileNetV3
    else:
        model = resnet.resnet18
    server = Server(trainer=ServerTrainer, model=model, algorithm=Algorithm)
    client = Client(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
