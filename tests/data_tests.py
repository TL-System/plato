"""
Testing datasources of Plato framework.
"""

import os
import unittest

os.environ['config_file'] = 'tests/TestsConfig/flickr30k_entities.yml'

import numpy as np
import torch

from plato.config import Config

from plato.datasources.flickr30k_entities import DataSource as f30ke_DataSource
from plato.datasources.referitgame import DataSource as refer_Datasource
from plato.datasources import registry as data_registry
from plato.samplers import registry as samplers_registry

from plato.samplers import modality_iid

from sampler_test_utils import define_sampler


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

    def assertDataSourceDefinition(self, data_source):
        """ Test whether the dataset can be correctly defined.
            This verifies:
            1.1- The datasource can be defined
            1.2- The raw data can be downloaded
            1.3- The correct data store structure can be set.

            2.1- The datasource can work with quantity-based samplers
            2.2- The defined data loader can load correct samples
            2.4- The visualization of samples are correct.
        """
        # Test 1
        self.utest_datasource = data_source

        # Test 2
        modality_sampler = modality_iid.Sampler(
            datasource=self.utest_datasource, client_id=self.client_id)
        testset = self.utest_datasource.get_test_set(modality_sampler.get())

        _ = testset.get_one_sample(sample_idx=10)

        batch_size = Config().trainer.batch_size
        # Define the sampler
        defined_sampler = define_sampler(Sampler=samplers_registry,
                                         dataset_source=self.utest_datasource,
                                         client_id=self.client_id,
                                         is_testing=True)
        testset_loader = torch.utils.data.DataLoader(
            dataset=testset,
            shuffle=False,
            batch_size=batch_size,
            sampler=defined_sampler.get())

        obtained_sample = next(iter(testset_loader))
        print("obtained_sample: ", obtained_sample)

        return True

    def test_f30ke_datasource(self):
        """ Test the flickr30k entities dataset. """
        self.utest_datasource = f30ke_DataSource()
        assert self.assertDataSourceDefinition(self.utest_datasource)

    # def test_ref_datasource(self):
    #     """ Test the flickr30k entities dataset. """
    #     # set the specific
    #     self.utest_datasource = refer_Datasource()
    #     assert self.assertDataSourceDefinition(self.utest_datasource)


if __name__ == '__main__':
    unittest.main()
