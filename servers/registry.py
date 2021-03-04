"""
A registry of all available servers.
"""

from servers import fedavg, fedavg_cs, fednova, mistnet, scaffold, rhythm, tempo, adaptive_sync

registered_servers = [
    fedavg.FedAvgServer, fedavg_cs.FedAvgCrossSiloServer,
    mistnet.MistNetServer, adaptive_sync.AdaptiveSyncServer,
    rhythm.RhythmServer, tempo.TempoServer, fednova.FedNovaServer,
    scaffold.ScaffoldServer
]


def get(server_type):
    """Get the server with the provided type."""
    server = None
    for registered_server in registered_servers:
        if registered_server.is_valid_server_type(server_type):
            server = registered_server.get_server()
            break

    if server is None:
        raise ValueError('No such server: {}'.format(server_type))

    return server
