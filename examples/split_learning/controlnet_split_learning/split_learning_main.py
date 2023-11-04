"""
Split learning with ControlNet.
"""
# pylint:disable=import-error
import os
import sys

sys.path.append(
    os.path.join(
        os.path.abspath(os.getcwd()), "examples/controlnet_split_learning/ControlNet"
    )
)
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "examples"))
# pylint:disable=wrong-import-position
from split_learning_client import Client
from split_learning_algorithm import Algorithm
from split_learning_trainer import Trainer
from split_learning_server import Server
from controlnet_datasource import DataSource
from OrgModel.model import ClientModel, ServerModel


def main():
    """A Plato federated learning training session using the split learning algorithm."""
    client = Client(
        model=ClientModel, datasource=DataSource, algorithm=Algorithm, trainer=Trainer
    )
    server = Server(
        model=ServerModel, datasource=DataSource, algorithm=Algorithm, trainer=Trainer
    )
    server.run(client)


if __name__ == "__main__":
    main()
