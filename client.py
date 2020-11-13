"""
Starting point for a Plato federated learning client.
"""

import asyncio
import logging
import websockets

from config import Config
from clients import SimpleClient


def main():
    """Starting a client to connect to the server via WebSockets."""
    __ = Config()
    client = SimpleClient()
    client.configure()

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(client.start_client())
    except websockets.ConnectionClosed:
        logging.info("Client #%s: connection to the server is closed.",
            client.client_id)


if __name__ == "__main__":
    main()
