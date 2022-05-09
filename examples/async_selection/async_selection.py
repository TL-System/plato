"""
A federated learning training session with asynchronous client selection.
"""
import os

os.environ['config_file'] = 

import async_selection_server

def main():
    """ A Plato federated learning training session using the FedSarah algorithm. """
    server = async_selection_server.Server()

    server.run()


if __name__ == "__main__":
    main()
