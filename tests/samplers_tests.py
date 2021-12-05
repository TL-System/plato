"""
Testing samplers in Plato framework.
"""

import os
import unittest

# os.environ['config_file'] = 'configs/Tests/distribution_noniid_sampler.yml'

# os.environ['config_file'] = 'configs/Tests/label_quantity_noniid_sampler.yml'

os.environ['config_file'] = 'configs/Tests/sample_quantity_noniid_sampler.yml'

import utils
import numpy as np

from plato.config import Config
from plato.datasources.cifar10 import DataSource
from plato.samplers import registry as samplers_registry


class SamplersTest(unittest.TestCase):
    """ Aiming to test the correcness of implemented samplers """
    def setUp(self):
        super().setUp()

        _ = Config()

        self.total_clients = Config().clients.total_clients
        self.utest_datasource = DataSource()

    def test_client_sampler_working(self):
        """ Test the client sampler works well, i.e., smoothly loading batches """
        clients_id = list(range(self.total_clients))
        client_id = np.random.choice(clients_id, 1)[0]
        utils.verify_working_correcness(Sampler=samplers_registry,
                                        dataset_source=self.utest_datasource,
                                        client_id=client_id,
                                        num_of_batches=10,
                                        batch_size=5,
                                        is_test_phase=False)

    def test_client_data_consistency(self):
        """ Test that the sampler always assignes same data distribution for one client
            It mainly verify:
             1- the assigned classes
             2- the assigned sample size for each class
             3- the samples index assigned to the client
        """
        assert utils.verify_client_data_correcness(
            Sampler=samplers_registry,
            dataset_source=self.utest_datasource,
            client_id=1,
            num_of_iterations=5,
            batch_size=5,
            is_presented=False)

    def test_clients_data_discrepancy(self):
        """ Test that different clients are assigned different local datasets """

        test_clients = list(range(10))
        dataset_classes = self.utest_datasource.classes()

        # filter the condition in the label quantity non-IID that
        #   each client is assigned full classes
        if Config().data.sampler == "label_quantity_noniid" \
            and Config().data.per_client_classes_size == len(dataset_classes):
            assert utils.verify_difference_between_clients(
                clients_id=test_clients,
                Sampler=samplers_registry,
                dataset_source=self.utest_datasource,
                num_of_batches=None,
                is_force_class_diff=False,
                batch_size=5,
                is_presented=False)
        else:
            assert utils.verify_difference_between_clients(
                clients_id=test_clients,
                Sampler=samplers_registry,
                dataset_source=self.utest_datasource,
                num_of_batches=None,
                is_force_class_diff=True,
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
                dataset_source=self.utest_datasource,
                required_classes_size=Config().data.per_client_classes_size,
                num_of_batches=None,
                batch_size=5,
                is_presented=False)


if __name__ == '__main__':
    unittest.main()
