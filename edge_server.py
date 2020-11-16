"""
Starting point for a Plato federated learning edge server.
"""

import asyncio
import logging
import websockets

from config import Config
from servers import EdgeServer


def main():
    """Starting an edge server to connect to the server via WebSockets."""
    __ = Config()
    edge = EdgeServer()
    edge.configure()

    # Will change to edge server's own address and port later
    start_edge_server = websockets.serve(edge.start_edge_server,
                    Config().server.address, Config().server.port,
                    ping_interval=None, max_size=2 ** 30)

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(start_edge_server)
    except websockets.ConnectionClosed:
        logging.info("Edge server #%s: connection to the central server is closed.",
            edge.edge_id)


if __name__ == "__main__":
    main()
