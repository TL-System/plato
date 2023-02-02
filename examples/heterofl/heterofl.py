"""
A federated learning training session using HeteroFL

Enmao Diao, Jie Ding, and Vahid Tarokh
“HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients,”
in ICLR, 2021.

Reference "https://github.com/dem123456789/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients".
"""

from mobilenetv3 import MobileNetV3
from plato.clients import simple

from heterofl_trainer import ServerTrainer


def main():
    """A Plato federated learning training session using the HeteroFL algorithm."""
    server = Server(trainer=ServerTrainer, model=MobileNetV3)
    client = simple.Client(model=MobileNetV3)
    server.run(client)


if __name__ == "__main__":
    main()
