""" An example for running Plato with custom clients. """
import asyncio
import logging
import os

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

os.environ['config_file'] = 'examples/dist_mistnet/mistnet_lenet5_client.yml'

from plato.clients import mistnet

class CustomClient(mistnet.Client):
    def __init__(self, model=None, datasource=None, trainer=None):
        super().__init__(model=model, datasource=datasource, trainer=trainer)
        logging.info("A customized client has been initialized.")

def main():
    """ 
    A Plato federated learning training session using a custom client.

    To run this example:
    python custom_client.py -i <client_id>
    """

    client = CustomClient()
    client.configure()
    asyncio.run(client.start_client())

if __name__ == "__main__":
    main()
