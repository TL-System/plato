"""
FjORD: Fair and Accurate Federated Learning under heterogeneous targets with Ordered Dropout

Samuel Horváth, Stefanos Laskaridis, Mario Almeida, Ilias Leontiadis, Stylianos Venieris, Nicholas Lane
in NeurIPS, 2021.
"""

from mobilenetv3 import MobileNetV3
import resnet
import vit

from fjord_client import Client
from fjord_server import Server
from fjord_algorithm import Algorithm
from fjord_trainer import ServerTrainer, ClientTrainer

from plato.config import Config


def main():
    """A Plato federated learning training session using the FjORD algorithm."""
    if "mobilenet" in Config().trainer.model_name:
        model = MobileNetV3
    elif "vit" in Config().trainer.model_name:
        model = vit.ViT
    else:
        model = resnet.resnet18
    server = Server(trainer=ServerTrainer, model=model, algorithm=Algorithm)
    client = Client(trainer=ClientTrainer, model=model, algorithm=Algorithm)
    server.run(client)


if __name__ == "__main__":
    main()
