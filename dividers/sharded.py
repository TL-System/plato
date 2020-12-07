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
        random.seed()
        self.__create_shards()

    def __create_shards(self):
        """Create all the shards (partitions) from the data."""
        # Extract the number of shards per client from the configuration
        per_client = Config().data.shard_per_client

        # Determine the correct total number of shards and the size of each shard
        total = Config().clients.total_clients * per_client
        shard_size = int(self.trainset_size / total)

        data = []
        for __, items in self.trainset.items():
            data.extend(items)

        shards = [
            data[(i * shard_size):((i + 1) * shard_size)] for i in range(total)
        ]
        random.shuffle(shards)

        self.shards = shards
        self.used = []

        logging.info('Created %s shards of size %s', len(shards), shard_size)

    def extract_shard(self):
        """Extract a shard from a list of shards."""
        shard = self.shards[0]
        self.used.append(shard)
        del self.shards[0]
        return shard

    def get_partition(self):
        """Get a partition for a client."""
        # Extract the number of shards per client
        per_client = Config().data.shard_per_client

        # Create data partition
        partition = []
        for _ in range(per_client):
            partition.extend(self.extract_shard())

        # Shuffle data partition
        random.shuffle(partition)

        return partition
