"""
A federated learning training session using Axiothea.

"""

import axiothea_server
import axiothea_client
import axiothea_edge


def main():
    """ A Plato federated learning training session using the axiothea algorithm. """
    client = axiothea_client.Client()
    server = axiothea_server.Server()
    edge_server = axiothea_server.Server
    edge_client = axiothea_edge.Client
    server.run(client, edge_server, edge_client)


if __name__ == "__main__":
    main()
