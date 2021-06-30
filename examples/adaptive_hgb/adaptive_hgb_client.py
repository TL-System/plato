#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A federated learning client with support for Adaptive gradient blending.

"""

from plato.config import Config

from plato.clients import simple

#   simple.Client
#   arguments: model=None, datasource=None, algorithm=None, trainer=None
#       One can either set these four parameters in the initialization or the client will
#   define these itself based on the configuration file

#   # The functions are required by the client
#   - configure: registe the trainer and the algorithm for this client
#   - load_data: obtain the trainset and testset from the datasoruce
#   - load_payload: the algorithm will be called to get the server's model to this client
#   - train: self.trainer.train operate the local training stage


class Client(simple.Client):
    """A federated learning client with support for Adaptive gradient blending.
    """
    def process_server_response(self, server_response):
        """Additional client-specific processing on the server response."""
        if 'sync_frequency' in server_response:
            Config().trainer = Config().trainer._replace(
                epochs=server_response['sync_frequency'])
