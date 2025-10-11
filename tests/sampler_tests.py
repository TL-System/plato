"""
Testing samplers in Plato framework.

How to run the tests:

For example, when you want to test the label_quantity_noniid_sampler.

 1 Uncomment the configuration file for the dataset you want to test
    # os.environ[
    'config_file'] = 'TestsConfig/label_quantity_noniid_sampler.yml'
    to   os.environ[
    'config_file'] = 'TestsConfig/label_quantity_noniid_sampler.yml'

 2 Run the following command in the root directory.
    python sampler_tests.py

"""

import os
import unittest

# os.environ['config_file'] = 'TestsConfig/distribution_noniid_sampler.yml'

# os.environ['config_file'] = 'TestsConfig/label_quantity_noniid_sampler.yml'

# os.environ[
#     'config_file'] = 'TestsConfig/mixed_label_quantity_noniid_sampler.yml'

os.environ["config_file"] = "TestsConfig/sample_quantity_noniid_sampler.yml"

import numpy as np
import sampler_test_utils

from plato.config import Config
from plato.datasources.cifar10 import DataSource
from plato.samplers import registry as samplers_registry


class SamplersTest(unittest.TestCase):
    """Testing the correctness of implemented samplers."""

    def setUp(self):
        super().setUp()

        _ = Config()

        self.total_clients = Config().clients.total_clients
        self.utest_datasource = DataSource()

    def test_client_sampler_working(self):
        """Testing whether the client sampler works well, i.e., smoothly loading batches."""
        clients_id = list(range(self.total_clients))
        client_id = np.random.choice(clients_id, 1)[0]

        sampler_test_utils.verify_working_correctness(
            Sampler=samplers_registry,
            dataset_source=self.utest_datasource,
            client_id=client_id,
            num_of_batches=10,
            batch_size=5,
            is_test_phase=False,
        )

    def test_client_data_consistency(self):
        """Testing whether the sampler always assigns the same data distribution to one client.

        It verifies:
         1 - the assigned classes
         2 - the assigned sample size for each class
         3 - the samples index assigned to the client
        """
        assert sampler_test_utils.verify_client_data_correctness(
            Sampler=samplers_registry,
            dataset_source=self.utest_datasource,
            client_id=1,
            num_of_iterations=5,
            batch_size=5,
            is_presented=False,
        )

    def test_clients_data_discrepancy(self):
        """Testing whether different clients are assigned different local datasets."""

        test_clients = list(range(10))
        dataset_classes = self.utest_datasource.classes()

        # Filter the condition in the label quantity non-IID that
        # each client is assigned full classes
        if (
            Config().data.sampler == "label_quantity_noniid"
            and Config().data.per_client_classes_size == len(dataset_classes)
        ):
            assert sampler_test_utils.verify_difference_between_clients(
                clients_id=test_clients,
                Sampler=samplers_registry,
                dataset_source=self.utest_datasource,
                num_of_batches=None,
                is_force_class_diff=False,
                batch_size=5,
                is_presented=False,
            )
        else:
            assert sampler_test_utils.verify_difference_between_clients(
                clients_id=test_clients,
                Sampler=samplers_registry,
                dataset_source=self.utest_datasource,
                num_of_batches=None,
                is_force_class_diff=True,
                batch_size=5,
                is_presented=False,
            )

    def test_clients_classes_size(self):
        """Testing whether the client contains a specific number of classes.
        This test is specifically designed for the label quantity non-iid sampler.
        """
        test_clients = list(range(10))

        if Config().data.sampler == "label_quantity_noniid":
            assert sampler_test_utils.verify_clients_fixed_classes(
                clients_id=test_clients,
                Sampler=samplers_registry,
                dataset_source=self.utest_datasource,
                required_classes_size=Config().data.per_client_classes_size,
                num_of_batches=None,
                batch_size=5,
                is_presented=False,
            )


if __name__ == "__main__":
    unittest.main()
