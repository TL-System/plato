""" An example for running Plato with custom clients. """
import asyncio
import os

os.environ['config_file'] = './mistnet_lenet5_client.yml'

from plato.clients import mistnet

def main():
    """
    A Plato federated learning client using the MistNet algorithm.

    To run this example:
    python mistnet_client.py -i <client_id>
    """

    client = mistnet.Client()
    client.configure()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.start_client())


if __name__ == "__main__":
    main()
