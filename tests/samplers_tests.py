"""
Testing samplers in Plato framework.
"""

import os
import unittest

# os.environ['config_file'] = 'configs/Kinetics/kinetics_mm.yml'

# os.environ['config_file'] = 'configs/Tests/distribution_noniid_sampler.yml'

os.environ['config_file'] = 'configs/Tests/label_quantity_noniid_sampler.yml'

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
        """ Test that the sampler always assignes same data distribution for one client """
        assert utils.verify_client_local_data_correcness(
            Sampler=samplers_registry,
            dataset_source=self.cifar10_datasource,
            client_id=1,
            num_of_iterations=5,
            batch_size=5,
            is_presented=False)

    def test_clients_data_discrepancy(self):
        """ Test that different clients are assigned different local datasets """

        test_clients = [0, 2, 4, 5]
        assert utils.verify_difference_between_clients(
            clients_id=test_clients,
            Sampler=samplers_registry,
            dataset_source=self.cifar10_datasource,
            num_of_batches=None,
            batch_size=5,
            is_presented=False)

    def test_clients_classes_size(self):
        """ Test whether the client contains specific number of classes
            This test is for the label quantity noniid sampler """
        test_clients = list(range(10))
        if Config().data.sampler == "label_quantity_noniid":
            assert utils.verify_clients_fixed_classes(
                clients_id=test_clients,
                Sampler=samplers_registry,
                dataset_source=self.cifar10_datasource,
                required_classes_size=Config().data.per_client_classes_size,
                num_of_batches=None,
                batch_size=5,
                is_presented=False)


if __name__ == '__main__':
    unittest.main()
