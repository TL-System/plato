"""
A federated learning training session using HeteroFL

Enmao Diao, Jie Ding, and Vahid Tarokh
“HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients,”
in ICLR, 2021.

Reference "https://github.com/dem123456789/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients".
"""

from mobilenetv3 import MobileNetV3
import resnet
from plato.config import Config
from heterofl_client import Client
from heterofl_server import Server
from heterofl_trainer import ServerTrainer


def main():
    """A Plato federated learning training session using the HeteroFL algorithm."""
    if "mobilenet" in Config().trainer.model_name:
        model=MobileNetV3
    else
        model=resnet.resnet152#check the paper, and choose the correct one
    server = Server(trainer=ServerTrainer, model=model)
    client = Client(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
