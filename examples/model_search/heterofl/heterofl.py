"""
A federated learning training session using HeteroFL

Enmao Diao, Jie Ding, and Vahid Tarokh
“HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients,”
in ICLR, 2021.

Reference "https://github.com/dem123456789/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients".
"""

import resnet
from heterofl_algorithm import Algorithm
from heterofl_client import create_client
from heterofl_server import Server
from heterofl_trainer import ServerTrainer

from plato.config import Config


def main() -> None:
    """A Plato federated learning training session using the HeteroFL algorithm."""
    model_name = Config().trainer.model_name
    if model_name == "resnet18":
        model = resnet.resnet18
    elif model_name in {"mobilenet", "mobilenet_v3_large", "mobilenet_v3_small"}:
        raise ValueError(
            "The custom HeteroFL MobileNetV3 model was retired. Historical "
            "source and restoration instructions: "
            "archives/retired/heterofl-mobilenetv3/README.md."
        )
    else:
        raise ValueError(f"No such HeteroFL model: {model_name}")
    server = Server(trainer=ServerTrainer, model=model, algorithm=Algorithm)
    client = create_client(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
