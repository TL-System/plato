"""
A federated learning training session using Rhythm.
"""
import os

os.environ['config_file'] = 'rhythm_MNIST_lenet5.yml'

import rhythm_server


def main():
    """ A Plato federated learning training session using the Rhythm algorithm. """
    server = rhythm_server.Server()
    server.run()


if __name__ == "__main__":
    main()
