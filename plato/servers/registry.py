"""
The registry for servers that contains framework-agnostic implementations on a federated
learning server.

Having a registry of all available classes is convenient for retrieving an instance based on a configuration at run-time.
"""
import logging
from collections import OrderedDict

from plato.servers import (
    fedavg,
    fedavg_cs,
    mistnet,
    rhythm,
    tempo,
    split_learning,
)

from plato.config import Config

registered_servers = OrderedDict([
    ('fedavg', fedavg.Server),
    ('fedavg_cross_silo', fedavg_cs.Server),
    ('mistnet', mistnet.Server),
    ('rhythm', rhythm.Server),
    ('tempo', tempo.Server),
    ('split_learning', split_learning.Server),
])


def get():
    """Get an instance of the server."""
    if hasattr(Config().server, 'type'):
        server_type = Config().server.type
    else:
        server_type = Config().algorithm.type

    if server_type in registered_servers:
        logging.info("Server: %s", server_type)
        registered_server = registered_servers[server_type]()
    else:
        raise ValueError('No such server: {}'.format(server_type))

    return registered_server
