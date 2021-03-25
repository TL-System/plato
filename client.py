"""
Starting point for a Plato federated learning client.
"""

from collections import OrderedDict
import asyncio
import logging
import websockets

from config import Config
import clients


def run(client_id, port):
    """Starting a client to connect to the server via WebSockets."""
    Config().args.id = client_id
    if port is not None:
        Config().args.port = port

    loop = asyncio.get_event_loop()
    coroutines = []
    client = None

    try:
        # If a server needs to be running concurrently
        if Config().is_edge_server():
            from servers import fedavg_cs, tempo, rhythm

            edge_servers = OrderedDict([
                ('fedavg_cross_silo', fedavg_cs.Server),
                ('tempo', tempo.Server),
                ('rhythm', rhythm.Server),
            ])

            Config().trainer = Config().trainer._replace(
                rounds=Config().algorithm.local_rounds)

            server = edge_servers[Config().server.type]()
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
                "fednova": clients.fednova.FedNovaClient,
                "tempo": clients.tempo.TempoClient,
                "scaffold": clients.scaffold.ScaffoldClient,
                "fedsarah": clients.fedsarah.FedSarahClient
            }[Config().clients.type]()
            logging.info("Starting a %s client.", Config().clients.type)
            client.configure()
            coroutines.append(client.start_client())

        loop.run_until_complete(asyncio.gather(*coroutines))

    except websockets.ConnectionClosed:
        logging.info("Client #%s: connection to the server is closed.",
                     client.client_id)


if __name__ == "__main__":
    __ = Config()
    run(Config().args.id, Config().args.port)
