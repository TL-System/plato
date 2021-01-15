"""
Starting point for a Plato federated learning client.
"""

import asyncio
import logging
import sys
import websockets

from config import Config
import clients
import servers


def main():
    """Starting a client to connect to the server via WebSockets."""
    __ = Config()

    loop = asyncio.get_event_loop()
    coroutines = []
    client = None

    try:
        # If a server needs to be running concurrently
        if Config().is_edge_server():
            Config().trainer = Config().trainer._replace(
                rounds=Config().cross_silo.rounds)

            if Config().rl:
                Config().server = Config().server._replace(
                    type=Config().rl.fl_server)

            server = {
                "fedavg": servers.fedavg.FedAvgServer,
                "fedavg_cross_silo": servers.fedavg_cs.FedAvgCrossSiloServer,
                "fedrl": servers.fedrl.FedRLServer,
                "mistnet": servers.mistnet.MistNetServer,
                "adaptive_sync": servers.adaptive_sync.AdaptiveSyncServer
            }[Config().server.type]()
            server.configure()

            client = clients.EdgeClient(server)
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
            client = {
                "simple": clients.SimpleClient,
                "mistnet": clients.MistNetClient,
                "adaptive_freezing":
                clients.adaptive_freezing.AdaptiveFreezingClient,
                "adaptive_sync": clients.adaptive_sync.AdaptiveSyncClient,
            }[Config().clients.type]()
            logging.info("Starting a %s client.", Config().clients.type)
            client.configure()
            coroutines.append(client.start_client())

        loop.run_until_complete(asyncio.gather(*coroutines))

    except websockets.ConnectionClosed:
        logging.info("Client #%s: connection to the server is closed.",
                     client.client_id)
        sys.exit()


if __name__ == "__main__":
    main()
