import os

os.environ['config_file'] = './config.yml'

from clients import simple
from servers import fedavg
from models import lenet5


def main():
    """A Plato federated learning training session using a custom model. """
    model = lenet5.Model()
    client = simple.Client(model)
    server = fedavg.Server(model)
    server.run(client)


if __name__ == "__main__":
    main()
