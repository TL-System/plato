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

from plato.samplers import modality_iid


class DatasetsTest(unittest.TestCase):
    """ Aiming to test the correcness of implemented samplers """
    def setUp(self):
        super().setUp()

        _ = Config()

        self.total_clients = Config().clients.total_clients

        # randomly client id
        clients_id = list(range(self.total_clients))
        self.client_id = np.random.choice(clients_id, 1)[0]

        # self.utest_datasource = data_registry.get(client_id=client_id)
        self.utest_datasource = None

    def test_datasource(self):
        """ Test whether the dataset can be correctly defined.
            This verify:
            1.1- The datasource can be defined
            1.2- The raw data can be downloaded
            1.3- The correct data store structure can be set.

            2.1- The datasource can work with quantity-based samplers
            2.2- The defined data loader can load correct samples
            2.4- The visualization of samples are correct.
        """
        # Test 1
        self.utest_datasource = DataSource()

        # Test 2
        modality_sampler = modality_iid.Sampler(
            datasource=self.utest_datasource, client_id=self.client_id)
        testset = self.utest_datasource.get_test_set(modality_sampler)

        testset.get_one_sample(sample_idx=10)
        # create the test dataset loader

        print(testset)


if __name__ == '__main__':
    unittest.main()
