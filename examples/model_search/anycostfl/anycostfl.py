"""
A federated learning training session using AnyCostFL

Peichun Li, Guoliang Cheng, Xumin Huang, Jiawen Kang, Rong Yu, Yuan Wu, Miao Pan
“AnycostFL: Efficient On-Demand Federated Learning over Heterogeneous Edge Device,”
in InfoCom 2023.

"""

from anycostfl_algorithm import Algorithm
from anycostfl_client import create_client
from anycostfl_server import Server
from anycostfl_trainer import ServerTrainer
from resnet import resnet18

from plato.config import Config


def main() -> None:
    """A Plato federated learning training session using the AnyCostFL algorithm."""
    model_name = Config().trainer.model_name
    if model_name == "resnet18":
        model = resnet18
    elif model_name == "vit":
        raise ValueError(
            "The AnyCostFL ViT model was retired. Historical source and "
            "restoration instructions: archives/retired/anycostfl-vit/README.md."
        )
    else:
        raise ValueError(f"No such AnyCostFL model: {model_name}")
    server = Server(model=model, algorithm=Algorithm, trainer=ServerTrainer)
    client = create_client(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
