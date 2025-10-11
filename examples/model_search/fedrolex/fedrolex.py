"""
A federated learning training session using FedRolexFL

Alam, Samiul and Liu, Luyang and Yan, Ming and Zhang, Mi
“FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction,”
in FedRolex NIPS2022.

"""

from fedrolex_algorithm import Algorithm
from fedrolex_client import Client
from fedrolex_server import Server
from fedrolex_trainer import ServerTrainer
from resnet import resnet18
from vit import ViT

from plato.config import Config


def main():
    """A Plato federated learning training session using the FedRolexFL algorithm."""
    if "resnet18" in Config().trainer.model_name:
        model = resnet18
    else:
        model = ViT
    server = Server(model=model, algorithm=Algorithm, trainer=ServerTrainer)
    client = Client(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
