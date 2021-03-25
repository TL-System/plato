"""
Starting point for a Plato federated learning client.
"""

from collections import OrderedDict
import asyncio
import logging
import websockets

from config import Config
from clients import registry as client_registry


def run(client_id, port, client=None):
    """Starting a client to connect to the server via WebSockets."""
    Config().args.id = client_id
    if port is not None:
        Config().args.port = port

    loop = asyncio.get_event_loop()
    coroutines = []

    try:
        # If a server needs to be running concurrently
        if Config().is_edge_server():
            from servers import fedavg_cs, tempo, rhythm
            from clients import edge

            edge_servers = OrderedDict([
                ('fedavg_cross_silo', fedavg_cs.Server),
                ('tempo', tempo.Server),
                ('rhythm', rhythm.Server),
            ])

            Config().trainer = Config().trainer._replace(
                rounds=Config().algorithm.local_rounds)

            server = edge_servers[Config().server.type]()
            server.configure()

            client = edge.Client(server)
            client.configure()
            coroutines.append(client.start_client())

            logging.info("Starting an edge server (client #%s) on port %s",
                         Config().args.id,
                         Config().args.port)
            start_server = websockets.serve(server.serve,
                                            Config().server.address,
                                            Config().args.port,
                                            ping_interval=None,
                                            max_size=2**30)

            coroutines.append(start_server)
        else:
            if client is None:
                client = client_registry.get()
                logging.info("Starting a %s client.", Config().clients.type)
            else:
                client.client_id = client_id
                logging.info("Starting a custom client #%s", client_id)

            client.configure()
            coroutines.append(client.start_client())

        loop.run_until_complete(asyncio.gather(*coroutines))

    except websockets.ConnectionClosed:
        logging.info("Client #%s: connection to the server is closed.",
                     client.client_id)


if __name__ == "__main__":
    __ = Config()
    run(Config().args.id, Config().args.port)
