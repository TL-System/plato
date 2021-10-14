"""
A federated learning training session using Tempo.
"""
import os

os.environ['config_file'] = 'tempo_MNIST_lenet5.yml'

import tempo_client
import tempo_server


def main():
    """ A Plato federated learning training session using the Tempo algorithm. """
    server = tempo_server.Server()
    client = tempo_client.Client()
    server.run(client)


if __name__ == "__main__":
    main()
