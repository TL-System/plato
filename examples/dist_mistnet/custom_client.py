""" An example for running Plato with custom clients. """
import asyncio
import logging
import os

os.environ['config_file'] = 'examples/dist_mistnet/mistnet_lenet5_client.yml'

from plato.clients import mistnet
from plato.utils import transmitter

class CustomClient(mistnet.Client):
    def __init__(self, model=None, datasource=None, trainer=None, transmitter=None):
        super().__init__(model=model, datasource=datasource, trainer=trainer, transmitter=transmitter)
        logging.info("A customized client has been initialized.")


def main():
    """
    A Plato federated learning training session using a custom client.

    To run this example:
    python custom_client.py -i <client_id>
    """

    trans = transmitter.S3Transmitter("https://obs.cn-south-1.myhuaweicloud.com", "EKPTZ0OPJC4SRAHPTZCA", "LiBjVWjbiVs37eiY9IdZ0OVnlBY4T3hBVgywaE9D", "plato")
    client = CustomClient(transmitter = trans)
    client.configure()
    asyncio.run(client.start_client())


if __name__ == "__main__":
    main()
