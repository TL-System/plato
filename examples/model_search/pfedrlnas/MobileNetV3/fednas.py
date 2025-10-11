"""
Implement new algorithm: personalized federarted NAS.

Reference Search Space: MobileNetv3.
https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf.
https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv3.html.
"""

from fednas_algorithm import ClientAlgorithm, ServerAlgorithm
from fednas_client import Client
from fednas_server import Server
from fednas_trainer import Trainer
from model.architect import Architect
from model.mobilenetv3_supernet import NasDynamicModel


def main():
    """
    A Plato federated learning training session using PerFedRLNAS.
    """
    supernet = NasDynamicModel
    client = Client(
        model=supernet,
        algorithm=ClientAlgorithm,
        trainer=Trainer,
    )
    server = Server(
        model=Architect,
        algorithm=ServerAlgorithm,
        trainer=Trainer,
    )
    server.run(client)


if __name__ == "__main__":
    main()
