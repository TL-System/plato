"""
A federated learning training session using AnyCostFL

Peichun Li, Guoliang Cheng, Xumin Huang, Jiawen Kang, Rong Yu, Yuan Wu, Miao Pan
“AnycostFL: Efficient On-Demand Federated Learning over Heterogeneous Edge Device,”
in InfoCom 2023.

"""

from anycostfl_algorithm import Algorithm
from anycostfl_client import Client
from anycostfl_server import Server
from anycostfl_trainer import ServerTrainer
from resnet import resnet18
from vit import ViT

from plato.config import Config


def main():
    """A Plato federated learning training session using the AnyCostFL algorithm."""
    if "resnet18" in Config().trainer.model_name:
        model = resnet18
    else:
        model = ViT
    server = Server(model=model, algorithm=Algorithm, trainer=ServerTrainer)
    client = Client(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
