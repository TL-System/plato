"""
Implement the client for the basic siamese method.

"""

import logging

from plato.config import Config
from plato.clients import simple
from plato.datasources import registry as datasources_registry
from plato.datasources import datawrapper_registry
from plato.samplers import registry as samplers_registry


class Client(simple.Client):

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.model_representation_weights_key = []

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        super().load_data()

        if hasattr(Config().data,
                   "data_wrapper") and Config().data.data_wrapper != None:

            self.trainset = datawrapper_registry.get(self.trainset)

        if Config().clients.do_test:
            if hasattr(Config().data,
                       "data_wrapper") and Config().data.data_wrapper != None:

                self.testset = datawrapper_registry.get(self.testset)
