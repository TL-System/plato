"""
FjORD: Fair and Accurate Federated Learning under heterogeneous targets with Ordered Dropout

Samuel Horváth, et at.
in NeurIPS, 2021.
"""

import resnet
import vit
from fjord_algorithm import Algorithm
from fjord_client import Client
from fjord_server import Server
from fjord_trainer import ClientTrainer, ServerTrainer

from plato.config import Config


def main():
    """A Plato federated learning training session using the FjORD algorithm."""
    if "vit" in Config().trainer.model_name:
        model = vit.ViT
    elif Config().trainer.model_name == "resnet18":
        model = resnet.resnet18
    elif Config().trainer.model_name == "resnet34":
        model = resnet.resnet34
    else:
        model = resnet.resnet152
    server = Server(trainer=ServerTrainer, model=model, algorithm=Algorithm)
    client = Client(trainer=ClientTrainer, model=model)
    server.run(client)


if __name__ == "__main__":
    main()
