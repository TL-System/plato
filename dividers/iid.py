import logging
import random

from dividers import base
from utils import dists

class IIDDivider(base.Divider):
    """Load IID data partitions."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)


    def extract(self, label, n):
        """Extract the data for a particular label."""
        if len(self.trainset[label]) > n:
            extracted = self.trainset[label][:n]  # Extract the data
            self.used[label].extend(extracted)  # Move data to used
            del self.trainset[label][:n]  # Remove from the trainset
            return extracted
        else:
            logging.warning('Insufficient data in label: %s', label)
            logging.warning('Reusing used data.')

            # Unmark data as used
            for label in self.labels:
                self.trainset[label].extend(self.used[label])
                self.used[label] = []

            # Extract replenished data
            return self.extract(label, n)


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
