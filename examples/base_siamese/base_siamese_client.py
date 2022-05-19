"""
Implement the client for base siamese method.

"""

import collections

from plato.config import Config
from plato.clients import simple
from plato.clients import base


class Client(simple.Client):

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)
        self.model_representation_weights_key = []