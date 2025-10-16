"""
The registry for servers that contains framework-agnostic implementations on a
federated learning server.

Having a registry of all available classes is convenient for retrieving an
instance based on a configuration at run-time.
"""

import logging

from plato.config import Config
from plato.servers import (
    fedavg,
    fedavg_cs,
    fedavg_gan,
    fedavg_mpc_additive,
    fedavg_mpc_shamir,
    fedavg_personalized,
    split_learning,
)

if hasattr(Config().server, "type") and Config().server.type == "fedavg_he":
    # FedAvg server with homomorphic encryption supports PyTorch only
    from plato.servers import fedavg_he

    registered_servers = {"fedavg_he": fedavg_he.Server}

else:
    registered_servers = {
        "fedavg": fedavg.Server,
        "fedavg_cross_silo": fedavg_cs.Server,
        "fedavg_gan": fedavg_gan.Server,
        "fedavg_personalized": fedavg_personalized.Server,
        "fedavg_mpc_additive": fedavg_mpc_additive.Server,
        "fedavg_mpc_shamir": fedavg_mpc_shamir.Server,
        "split_learning": split_learning.Server,
    }


def get(model=None, algorithm=None, trainer=None):
    """Get an instance of the server."""
    if hasattr(Config().server, "type"):
        server_type = Config().server.type
    else:
        server_type = Config().algorithm.type

    if server_type in registered_servers:
        logging.info("Server: %s", server_type)
        return registered_servers[server_type](
            model=model, algorithm=algorithm, trainer=trainer
        )
    else:
        raise ValueError(f"No such server: {server_type}")
