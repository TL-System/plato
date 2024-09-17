"""
Split learning with ControlNet.
"""
# pylint:disable=import-error
import os
import sys

sys.path.append(
    os.path.join(
        os.path.abspath(os.getcwd()),
        "examples/split_learning/controlnet_split_learning/ControlNet",
    )
)
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "examples"))
# pylint:disable=wrong-import-position
from plato.servers.split_learning import Server
from plato.clients.split_learning import Client
from split_learning_trainer import Trainer
from controlnet_datasource import DataSource
from ControlNetSplitLearning.model import ClientModel, ServerModel


def main():
    """A Plato federated learning training session using the split learning algorithm."""
    client = Client(model=ClientModel, datasource=DataSource, trainer=Trainer)
    server = Server(model=ServerModel, datasource=DataSource, trainer=Trainer)
    server.run(client)


if __name__ == "__main__":
    main()
