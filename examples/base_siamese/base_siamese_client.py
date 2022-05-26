"""
Implement the client for the basic siamese method.

"""

from plato.config import Config
from plato.clients import simple

from plato.datasources import datawrapper_registry


class Client(simple.Client):

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

    def load_data(self) -> None:
        """Generating data and loading them onto this client."""
        super().load_data()

        if hasattr(Config().data,
                   "data_wrapper") and Config().data.data_wrapper != None:

            self.trainset = datawrapper_registry.get(self.trainset)
