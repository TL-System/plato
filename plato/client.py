"""
Starting point for a Plato federated learning client.
"""

import asyncio
import logging
import os

from plato.clients import registry as client_registry
from plato.config import Config


def run(client_id, port, client=None, edge_server=None, edge_client=None):
    """Starting a client to connect to the server."""
    Config().args.id = client_id
    if port is not None:
        Config().args.port = port

    # If a server needs to be running concurrently
    if Config().is_edge_server():
        Config().trainer = Config().trainer._replace(
            rounds=Config().algorithm.local_rounds)

        if edge_server is None:
            from plato.clients import edge
            from plato.servers import fedavg_cs
            server = fedavg_cs.Server()
            client = edge.Client(server)
        else:
            # A custom edge server
            server = edge_server()
            client = edge_client(server)

        server.configure()
        client.configure()

        logging.info("Starting an edge server as client #%d on port %d",
                     Config().args.id,
                     Config().args.port)
        asyncio.ensure_future(client.start_client())

        logging.info("Starting an edge server as server #%d on port %d",
                     os.getpid(),
                     Config().args.port)
        server.start(port=Config().args.port)

    else:
        if client is None:
            client = client_registry.get()
            logging.info("Starting a %s client #%d.",
                         Config().clients.type, client_id)
        else:
            client.client_id = client_id
            logging.info("Starting a custom client #%d", client_id)

        client.configure()
        asyncio.run(client.start_client())


if __name__ == "__main__":
    run(Config().args.id, Config().args.port)
