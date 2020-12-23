"""
Starting point for a Plato federated learning client.
"""

import asyncio
import logging
import sys
import websockets

from config import Config
from clients import SimpleClient, EdgeClient
import servers


def main():
    """Starting a client to connect to the server via WebSockets."""
    __ = Config()

    loop = asyncio.get_event_loop()
    coroutines = []

    try:
        # If a server needs to be running concurrently
        if Config().args.port:
            if Config().rl:
                server = {
                    "fedavg": servers.fedavg.FedAvgServer
                }[Config().rl.fl_server]()
            else:
                server = {
                    "fedavg": servers.fedavg.FedAvgServer
                }[Config().server.type]()
            server.configure()

            client = EdgeClient(server)
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
            client = SimpleClient()
            client.configure()
            coroutines.append(client.start_client())

        loop.run_until_complete(asyncio.gather(*coroutines))

    except websockets.ConnectionClosed:
        logging.info("Client #%s: connection to the server is closed.",
                     client.client_id)
        sys.exit()


if __name__ == "__main__":
    main()
