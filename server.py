"""
Starting point for a Plato federated learning server.
"""

import asyncio
import time
import logging
import websockets

from config import Config
import servers


def start_server_and_clients(server):
    """Starting a WebSockets server and its clients."""

    logging.info("Starting a server on port %s...", Config().server.port)
    start_server = websockets.serve(server.serve,
                                    Config().server.address,
                                    Config().server.port,
                                    ping_interval=None,
                                    max_size=2**30)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)

    if Config().cross_silo:
        server.start_clients(as_server=True)
        # Allowing some time for the edge servers to start
        time.sleep(5)

    server.start_clients()


def main():
    """Configure and start a WebSockets server."""

    __ = Config()

    server = {"fedavg": servers.fedavg.FedAvgServer}[Config().server.type]()
    server.configure()
    start_server_and_clients(server)

    loop = asyncio.get_event_loop()
    loop.run_forever()


if __name__ == "__main__":
    main()
