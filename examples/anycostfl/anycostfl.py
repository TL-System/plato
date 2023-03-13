"""
A federated learning training session using AnyCostFL

Peichun Li, Guoliang Cheng, Xumin Huang, Jiawen Kang, Rong Yu, Yuan Wu, Miao Pan
“AnycostFL: Efficient On-Demand Federated Learning over Heterogeneous Edge Device,”
in InfoCom 2023.

"""

from resnet import resnet34, resnet152
from vit import ViT
from vgg import VGG

from plato.config import Config
from anycostfl_client import Client
from anycostfl_server import Server
from anycostfl_algorithm import Algorithm
from anycostfl_trainer import ServerTrainer


def main():
    """A Plato federated learning training session using the AnyCostFL algorithm."""
    if "resnet34" in Config().trainer.model_name:
        model = resnet34
    elif "resnet152" in Config().trainer.model_name:
        model = resnet152
    elif "vgg" in Config().trainer.model_name:
        model = VGG
    else:
        model = ViT
    server = Server(model=model, algorithm=Algorithm, trainer=ServerTrainer)
    client = Client(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
