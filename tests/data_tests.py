"""
Testing multi-modal datasources of Plato framework.
"""

import os
import unittest

os.environ['config_file'] = 'tests/TestsConfig/flickr30k_entities.yml'

# os.environ['config_file'] = 'tests/TestsConfig/coco.yml'

# os.environ['config_file'] = 'tests/TestsConfig/referitgame.yml'

# os.environ['config_file'] = 'tests/TestsConfig/kinetics.yml'

# os.environ['config_file'] = 'tests/TestsConfig/gym.yml'

# Note: the plato will search the dir './config' for Pipeline and other configuration files
#   directly by default. This is achieved by the code in line 83 of 'config.py'

import numpy as np
import torch

from plato.config import Config

from plato.datasources.flickr30k_entities import DataSource as f30ke_DataSource
# from plato.datasources.referitgame import DataSource as refer_Datasource
# from plato.datasources.coco import DataSource as coco_Datasource
# from plato.datasources.kinetics import DataSource as kinetics_Datasource
# from plato.datasources.gym import DataSource as GymDataSource
from plato.samplers import registry as samplers_registry

from plato.samplers import modality_iid

from sampler_test_utils import define_sampler


class DatasetsTest(unittest.TestCase):
    """ Aiming to test the correcness of implemented samplers and datasets """
    def setUp(self):
        super().setUp()

        _ = Config()

        self.total_clients = Config().clients.total_clients

        # randomly client id
        clients_id = list(range(self.total_clients))
        self.client_id = np.random.choice(clients_id, 1)[0]

        # the datasource being tested
        self.utest_datasource = None

    def assertDataSourceDefinition(self, data_source):
        """ Test whether the dataset can be correctly defined.
            This verifies:
            1.1- The datasource can be defined
            1.2- The raw data can be downloaded
            1.3- The correct data store structure can be set.

            2.1- The datasource can work with defined samplers
            2.2- The defined data loader can load correct samples
            2.4- The visualization of samples are correct.
        """
        # Test 1
        self.utest_datasource = data_source

        # Test 2
        modality_sampler = modality_iid.Sampler(
            datasource=self.utest_datasource, client_id=self.client_id)
        test_dataset = self.utest_datasource.get_test_set(
            modality_sampler.get())

        _ = test_dataset.get_one_multimodal_sample(sample_idx=0)
        _ = test_dataset[0]

        batch_size = Config().trainer.batch_size
        # define the sampler
        defined_sampler = define_sampler(Sampler=samplers_registry,
                                         dataset_source=self.utest_datasource,
                                         client_id=self.client_id,
                                         is_testing=True)
        testset_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=batch_size,
            sampler=defined_sampler.get())

        obtained_mm_sample = next(iter(testset_loader))
        print("obtained_sample: ", obtained_mm_sample)

        return True

    def test_f30ke_datasource(self):
        """ Test the flickr30k entities dataset. """
        self.utest_datasource = f30ke_DataSource()
        assert self.assertDataSourceDefinition(self.utest_datasource)

    # def test_coco_datasource(self):
    #     """ Test the MSCOCO dataset. """
    #     # set the specific

    #     self.utest_datasource = coco_Datasource()
    #     # assert self.assertDataSourceDefinition(self.utest_datasource)

    # def test_ref_datasource(self):
    #     """ Test the ReferItGmae dataset. """
    #     # set the specific

    #     self.utest_datasource = refer_Datasource()
    #     assert self.assertDataSourceDefinition(self.utest_datasource)

    # def test_kinetics_datasource(self):
    #     """ Test the kinetics700 dataset. """
    #     # set the specific

    #     self.utest_datasource = kinetics_Datasource()
    #     kinetics_train_dataset = self.utest_datasource.get_train_set(
    #         modality_sampler=None)

    #     iter_data = kinetics_train_dataset[0]
    #     print("rgb: ")
    #     print(iter_data["rgb"]["imgs"].shape)
    #     print(iter_data["rgb"]["label"].shape)

    #     assert self.assertDataSourceDefinition(self.utest_datasource)

    # def test_gym_datasource(self):
    #     """ Test the Gym dataset. """
    #     # set the specific

    #     self.utest_datasource = GymDataSource()
    #     assert self.assertDataSourceDefinition(self.utest_datasource)


if __name__ == '__main__':
    unittest.main()
