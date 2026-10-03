"""
A federated learning training session using FedRolexFL

Alam, Samiul and Liu, Luyang and Yan, Ming and Zhang, Mi
“FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction,”
in FedRolex NIPS2022.

"""

from fedrolex_algorithm import Algorithm
from fedrolex_client import create_client
from fedrolex_server import Server
from fedrolex_trainer import ServerTrainer
from resnet import resnet18

from plato.config import Config


def main() -> None:
    """A Plato federated learning training session using the FedRolexFL algorithm."""
    model_name = Config().trainer.model_name
    if model_name == "resnet18":
        model = resnet18
    elif model_name == "vit":
        raise ValueError(
            "The FedRolex ViT model was retired. Historical source and "
            "restoration instructions: archives/retired/fedrolex-vit/README.md."
        )
    else:
        raise ValueError(f"No such FedRolex model: {model_name}")
    server = Server(model=model, algorithm=Algorithm, trainer=ServerTrainer)
    client = create_client(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
