import logging
import random

from dividers import base
from utils import dists

class IIDDivider(base.Divider):
    """Load IID data partitions."""

    def __init__(self, dataset):
        super().__init__(dataset)
        random.seed()


    def get_partition(self, partition_size):
        """Get a partition that is uniform across all the labels."""
        # Use uniform distribution
        dist, __ = dists.uniform(partition_size, len(self.labels))

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition


    def get_testset(self):
        """Return the entire testset."""
        return self.testset
