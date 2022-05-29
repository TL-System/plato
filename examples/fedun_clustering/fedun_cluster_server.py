"""
A customized server for the federated unlearning baseline clustering algorithm.

"""
import logging
import os
import time

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """ A federated unlearning server that implements the federated unlearning 
    baseline clustering algorithm.
    The work pipeline of the server is as below.
    1. The server divided total clients randomly into N clusters.
    2. The clients then do the training.
    3. Aggregate the updates in clusters.
    4. Do the global aggregation(aggregate from all clusters) at the target 
    round and stop training.
    """
