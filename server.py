"""
Starting point for a Plato federated learning server.
"""

import asyncio
import websockets

from config import Config
import servers

def main():
    """Starting a WebSockets server."""
    __ = Config()

    server = {
        "fedavg": servers.fedavg.FedAvgServer,
        "fedcs": servers.fedcs.CrossSiloServer
    }[Config().training.server]()

    if Config().edges is not None:
        server.start_edge_servers()

    server.start_clients()

    start_server = websockets.serve(server.serve,
                    Config().server.address, Config().server.port,
                    ping_interval=None, max_size=2 ** 30)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()


if __name__ == "__main__":
    main()
