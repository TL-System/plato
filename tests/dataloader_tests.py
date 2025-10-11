"""Unit tests for data loaders."""

import math
import os
import unittest
from typing import List, Type

import torch

os.environ["config_file"] = "config.yml"

import numpy as np

from plato.config import Config
from plato.datasources import registry as datasource_registry
from plato.samplers import registry as samplers_registry
from plato.utils import data_loaders


class DataLoadersTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        __ = Config()

        # set client id
        self.client1_id = 10
        self.client2_id = 20
        self.client3_id = 39

        self.utest_datasource = datasource_registry.get()

        self.client1_batch_size = 8
        self.client2_batch_size = 64
        self.client3_batch_size = 128

        # get partition size of each client
        self.partition_size = Config().data.partition_size

        # as plato's sample will pad the #samples based on the
        # batch_size to make #samples % batch_size == 0
        # thus, the data length should the upper bound
        self.client1_data_length = math.ceil(
            self.partition_size / self.client1_batch_size
        )
        self.client2_data_length = math.ceil(
            self.partition_size / self.client2_batch_size
        )
        self.client3_data_length = math.ceil(
            self.partition_size / self.client3_batch_size
        )

    def define_client_dataloader(self, client_id: int, testing: bool, batch_size: int):
        """Define the client dataloader for the input client."""
        sampler = samplers_registry.get(
            datasource=self.utest_datasource, client_id=client_id, testing=testing
        )
        dataset = (
            self.utest_datasource.get_test_set()
            if testing
            else self.utest_datasource.get_train_set()
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler.get(),
        )
        return data_loader

    def check_normal_dataloader(
        self,
        data_loader: Type[torch.utils.data.DataLoader],
        target_batch_size: int,
        target_loader_length: int,
    ):
        """A use case of iterating through a normal dataloader."""
        num_batches = 0
        for batch_id, batch_samples in enumerate(data_loader):
            num_batches += 1

            if batch_id % 40 == 0:
                batch_data, _ = batch_samples
                batch_size = batch_data.shape[0]
                self.assertEqual(batch_size, target_batch_size)

        self.assertEqual(num_batches, target_loader_length)

    def check_parallel_dataloader(
        self,
        data_loader: Type[torch.utils.data.DataLoader],
        target_batch_sizes: List[int],
        target_loaders_length: List[int],
    ):
        """A use case of iterating through a parallel dataloader."""
        num_batches = 0
        for batch_id, loaders_batch in enumerate(data_loader):
            if batch_id % 50 == 0:
                self.assertIsInstance(loaders_batch, list)
                self.assertEqual(len(target_batch_sizes), len(loaders_batch))

                for loader_idx, batch_samples in enumerate(loaders_batch):
                    data, _ = batch_samples
                    batch_size = data.shape[0]
                    self.assertEqual(batch_size, target_batch_sizes[loader_idx])

            num_batches += 1

        minimum_loader_length = min(target_loaders_length)
        self.assertEqual(minimum_loader_length, num_batches)

    def check_sequence_dataloader(
        self,
        data_loader: Type[torch.utils.data.DataLoader],
        target_batch_sizes: List[int],
        target_loaders_length: List[int],
    ):
        """A use case of iterating through my_loader."""
        num_batches = 0
        loaders_batch_bound = np.cumsum(target_loaders_length, axis=0)
        for batch_id, batch_samples in enumerate(data_loader):
            cur_loader_idx = np.digitize(batch_id, loaders_batch_bound)
            if batch_id % 40 == 0:
                batch_data, _ = batch_samples
                batch_size = batch_data.shape[0]
                target_batch_size = target_batch_sizes[cur_loader_idx]

                self.assertEqual(batch_size, target_batch_size)

            num_batches += 1

        total_length = sum(target_loaders_length)
        self.assertEqual(num_batches, total_length)

    def test_train_dataloader(self):
        client1_dataloader = self.define_client_dataloader(
            client_id=self.client1_id, testing=True, batch_size=self.client1_batch_size
        )
        client2_dataloader = self.define_client_dataloader(
            client_id=self.client2_id, testing=True, batch_size=self.client2_batch_size
        )
        client3_dataloader = self.define_client_dataloader(
            client_id=self.client3_id, testing=True, batch_size=self.client3_batch_size
        )
        print("client1_dataloader: ", client1_dataloader)
        # define different data loaders for comprehensive tests
        # adding the 'None' here to test the specific case
        parallel_loader23 = data_loaders.ParallelDataLoader(
            [client2_dataloader, client3_dataloader, None]
        )
        parallel_loader123 = data_loaders.ParallelDataLoader(
            [client1_dataloader, client2_dataloader, client3_dataloader]
        )

        sequence_loader23 = data_loaders.SequentialDataLoader(
            [client2_dataloader, client3_dataloader, None]
        )
        sequence_loader123 = data_loaders.SequentialDataLoader(
            [client1_dataloader, client2_dataloader, client3_dataloader, None]
        )

        # check the normal data loader
        self.check_normal_dataloader(
            client1_dataloader,
            target_batch_size=self.client1_batch_size,
            target_loader_length=self.client1_data_length,
        )
        self.check_normal_dataloader(
            client2_dataloader,
            target_batch_size=self.client2_batch_size,
            target_loader_length=self.client2_data_length,
        )
        self.check_normal_dataloader(
            client3_dataloader,
            target_batch_size=self.client3_batch_size,
            target_loader_length=self.client3_data_length,
        )

        # check the paraller data loader
        self.check_parallel_dataloader(
            parallel_loader23,
            target_batch_sizes=[self.client2_batch_size, self.client3_batch_size],
            target_loaders_length=[self.client2_data_length, self.client3_data_length],
        )

        self.check_parallel_dataloader(
            parallel_loader123,
            target_batch_sizes=[
                self.client1_batch_size,
                self.client2_batch_size,
                self.client3_batch_size,
            ],
            target_loaders_length=[
                self.client1_data_length,
                self.client2_data_length,
                self.client3_data_length,
            ],
        )

        # check the sequence data loader
        self.check_sequence_dataloader(
            sequence_loader23,
            target_batch_sizes=[self.client2_batch_size, self.client3_batch_size],
            target_loaders_length=[self.client2_data_length, self.client3_data_length],
        )

        self.check_sequence_dataloader(
            sequence_loader123,
            target_batch_sizes=[
                self.client1_batch_size,
                self.client2_batch_size,
                self.client3_batch_size,
            ],
            target_loaders_length=[
                self.client1_data_length,
                self.client2_data_length,
                self.client3_data_length,
            ],
        )


if __name__ == "__main__":
    unittest.main()
