"""
Testing datasources of Plato framework.
"""

import os
import unittest

os.environ['config_file'] = 'tests/TestsConfig/flickr30k_entities.yml'

import numpy as np

from plato.config import Config

from plato.datasources.flickr30k_entities import DataSource
from plato.datasources import registry as data_registry
from plato.samplers import registry as samplers_registry


class DatasetsTest(unittest.TestCase):
    """ Aiming to test the correcness of implemented samplers """
    def setUp(self):
        super().setUp()

        _ = Config()

        self.total_clients = Config().clients.total_clients

        # randomly client id
        clients_id = list(range(self.total_clients))
        client_id = np.random.choice(clients_id, 1)[0]

        # self.utest_datasource = data_registry.get(client_id=client_id)
        self.utest_datasource = None

    def test_data_define(self):
        """ Test whether the dataset can be correctly defined. """
        self.utest_datasource = DataSource()


if __name__ == '__main__':
    unittest.main()
