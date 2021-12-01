"""
Testing a federated learning configuration.
"""

import os
import unittest

os.environ['config_file'] = 'configs/Kinetics/kinetics_mm.yml'

import utils

from plato.config import Config
from plato.datasources.cifar10 import DataSource
from plato.samplers import registry as samplers_registry


class SamplersTest(unittest.TestCase):
    """ Aiming to test the correcness of implemented samplers """
    def setUp(self):
        super().setUp()

        _ = Config()

        self.cifar10_datasource = DataSource()

    def test_client_data_consistency(self):
        """ Test that the sampler always assignes same data distribution of one client """
        assert utils.verify_client_local_data_correcness(
            Sampler=samplers_registry,
            dataset_source=self.cifar10_datasource,
            client_id=1,
            num_of_iterations=5,
            batch_size=5,
            is_presented=False)

    def test_clients_data_discrepancy(self):
        """ Test that different clients are assigned different local datasets """

        selected_clients = [0, 2, 4, 5]
        assert utils.verify_difference_between_clients(
            clients_id=selected_clients,
            Sampler=samplers_registry,
            dataset_source=self.cifar10_datasource,
            num_of_batches=None,
            batch_size=5,
            is_presented=False)


if __name__ == '__main__':
    unittest.main()
