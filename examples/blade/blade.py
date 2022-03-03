"""
A federated learning training session using Blade.

"""

import blade_server
import blade_client
import blade_edge


def main():
    """ A Plato federated learning training session using the Blade algorithm. """
    client = blade_client.Client()
    server = blade_server.Server()
    edge_server = blade_server.Server
    edge_client = blade_edge.Client
    server.run(client, edge_server, edge_client)


if __name__ == "__main__":
    main()
