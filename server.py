"""
Starting point for a Plato federated learning server.
"""

import asyncio
import time
import logging
import websockets

from config import Config
import servers


def main():
    """Starting a WebSockets server."""
    __ = Config()

    # Remove the running trainers table from previous runs
    Config().cursor.execute("DROP TABLE IF EXISTS trainers")

    server = {
        "fedavg": servers.fedavg.FedAvgServer,
        "fedavg_cross_silo": servers.fedavg_cs.FedAvgCrossSiloServer,
        "mistnet": servers.mistnet.MistNetServer,
        "adaptive_sync": servers.adaptive_sync.AdaptiveSyncServer,
        "rhythm": servers.rhythm.RhythmServer,
        "tempo": servers.tempo.TempoServer
    }[Config().algorithm.type]()
    server.configure()

    logging.info("Starting a server on port %s", Config().server.port)
    start_server = websockets.serve(server.serve,
                                    Config().server.address,
                                    Config().server.port,
                                    ping_interval=None,
                                    max_size=2**30)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)

    if Config().is_central_server():
        # In cross-silo FL, the central server lets edge servers start first
        # Then starts their clients
        server.start_clients(as_server=True)
        # Allowing some time for the edge servers to start
        time.sleep(5)

    server.start_clients()
    loop.run_forever()


if __name__ == "__main__":
    main()
