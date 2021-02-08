"""
Divide data into partitions with sharding, where data is to be horizontally partitioned.
"""

import random
import logging

from dividers import base
from config import Config


class ShardedDivider(base.Divider):
    """
    Divide data into partitions with sharding, where data is to be horizontally partitioned.
    """
    def __init__(self, dataset):
        super().__init__(dataset)
        self.shards = None
        self.__create_shards()

    def __create_shards(self):
        """Create all the shards (partitions) from the data."""
        # Extract the number of shards per client from the configuration
        per_client = Config().data.shard_per_client

        # Determine the correct total number of shards and the size of each shard
        total = Config().clients.total_clients * per_client
        shard_size = int(self.trainset_size() / total)

        data = []
        for __, items in self.trainset.items():
            data.extend(items)

        shards = [
            data[(i * shard_size):((i + 1) * shard_size)] for i in range(total)
        ]

        self.shards = shards

        logging.info("Created %s shards of size %s", len(shards), shard_size)

    def get_partition(self, client_id):
        """Get a partition for a client."""
        # Extract the number of shards per client
        per_client = Config().data.shard_per_client

        # The index of shard that loaded to this client
        shard_id_list = [(int(client_id) - 1) * per_client + i
                         for i in range(per_client)]

        # Create data partition
        partition = []
        for shard_id in shard_id_list:
            partition.extend(self.shards[shard_id])

        # Shuffle data partition
        random.shuffle(partition)

        return partition

    def trainset_size(self):
        """Return the size of the whole dataset."""
        trainset_size = 0
        for label in self.trainset:
            trainset_size += len(self.trainset[label])
        return trainset_size
