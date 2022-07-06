"""
A federated learning training session with asynchronous client selection.
"""
import os

os.environ['config_file'] = './async_mnist_lenet5.yml'

import async_selection_server
#import async_selection_client


def main():
    """ A Plato federated learning training session using the FedSarah algorithm. """
    server = async_selection_server.Server()
    #client = async_selection_client.Client()

    server.run()


if __name__ == "__main__":
    main()
