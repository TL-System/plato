import logging
import os

from torch import nn

# os.environ['config_file'] = 'examples/configs/server.yml'

from plato.servers import fedavg
from plato.utils import transmitter

class CustomServer(fedavg.Server):
    """ A custom federated learning server. """
    def __init__(self, model=None, trainer=None, transmitter=None):
        super().__init__(model, trainer, transmitter=transmitter)
        logging.info("A custom server has been initialized.")


def main():
    """ A Plato federated learning training session using a custom model. """
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    
    trans = transmitter.S3Transmitter("https://obs.cn-south-1.myhuaweicloud.com", "EKPTZ0OPJC4SRAHPTZCA", "LiBjVWjbiVs37eiY9IdZ0OVnlBY4T3hBVgywaE9D", "plato")
    # server = CustomServer(model=model)
    server = CustomServer(transmitter = trans)
    server.run()

if __name__ == "__main__":
    main()
