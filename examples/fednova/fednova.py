import os

os.environ['config_file'] = 'fednova_MNIST_lenet5.yml'

import fednova_client
import fednova_server

def main():
    """ A Plato federated learning training session using the FedNova algorithm. """
    client = fednova_client.Client()
    server = fednova_server.Server()
    server.run(client)

if __name__ == "__main__":
    main()
