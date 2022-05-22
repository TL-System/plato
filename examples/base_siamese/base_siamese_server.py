"""
Implement the server for base siamese method.

"""

from plato.servers import fedavg
from plato.datasources import datawrapper_registry
from plato.config import Config


class Server(fedavg.Server):

    def __init__(self, model=None, algorithm=None, trainer=None):
        super().__init__(model, algorithm, trainer)

    def configure(self):
        super().configure()

        if not (hasattr(Config().server, 'do_test')
                and not Config().server.do_test):
            self.testset = datawrapper_registry.get(self.testset)