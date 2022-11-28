"""Unit tests for the data loaders."""
import os
import unittest

import torch

from plato.config import Config
from plato.datasources.cifar10 import DataSource
from plato.utils import data_loaders_wrapper
from plato.samplers import registry as samplers_registry


class DataLoadersTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        __ = Config()

        # set a randomly constant value
        # as the client id.
        self.client1_id = 10
        self.client2_id = 20
        self.client3_id = 39
        self.utest_datasource = DataSource()

    def define_client_dataloader(self, client_id, testing, batch_size):
        """ Define the client dataloader for the input client. """
        sampler = samplers_registry.get(
            datasource=self.utest_datasource, client_id=client_id, testing=testing
        )
        dataset = self.utest_datasource.get_test_set() if testing else self.utest_datasource.get_train_set()
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler.get(),
        )
        return data_loader

    def check_normal_dataloader(self, data_loader):
        """A use case of iterating through a normal dataloader."""
        for i, b1 in enumerate(data_loader):
            data, target = b1
            if i in [100, 200]:
                print(type(data), data.size(), type(target), target.size())
        print('num of batches: {}'.format(i + 1))

    def check_combined_dataloader(self, data_loader):
        """A use case of iterating through my_loader."""
        stopped_batches = 0
        for i, batches in enumerate(data_loader):
            if i in [10, 20]:
                for j, b in enumerate(batches):
                    data, target = b
                    print(j + 1, type(data), data.size(), type(target),
                        target.size())
            stopped_batches = i

        print('stopped_batches: {}'.format(stopped_batches + 1))

    def check_sequence_dataloader(self, data_loader):
        """A use case of iterating through my_loader."""
        stopped_batches = 0
        for batch_id, batch in enumerate(data_loader):
            if batch_id % 100 == 0:
                data, target = batch
                print(batch_id + 1, type(data), data.size(), type(target),
                    target.size())

            stopped_batches = batch_id

        print('stopped_batches: {}'.format(stopped_batches + 1))

    def test_train_dataloader(self):
        client1_dataloader = self.define_client_dataloader(client_id=self.client1_id, testing=True, batch_size=8)
        client2_dataloader = self.define_client_dataloader(client_id=self.client2_id, testing=True, batch_size=64)
        client3_dataloader = self.define_client_dataloader(client_id=self.client3_id, testing=True, batch_size=128)

        parallel_loader = data_loaders_wrapper.CombinedBatchesLoader([client1_dataloader, client2_dataloader, client3_dataloader])
        sequence_loader = data_loaders_wrapper.StreamBatchesLoader([client1_dataloader, client2_dataloader, client3_dataloader, None])



    def test_test_dataloader(self):
        client1_dataloader = self.define_client_dataloader(client_id=self.client1_id, testing=False)
        client2_dataloader = self.define_client_dataloader(client_id=self.client2_id, testing=False)
        client3_dataloader = self.define_client_dataloader(client_id=self.client3_id, testing=False)

        parallel_loader = data_loaders_wrapper.CombinedBatchesLoader([client1_dataloader, client2_dataloader, client3_dataloader])
        sequence_loader = data_loaders_wrapper.StreamBatchesLoader([client1_dataloader, client2_dataloader, client3_dataloader, None])




if __name__ == "__main__":
    unittest.main()
