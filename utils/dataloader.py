"""
Parent class for data loaders.
"""

import logging
import random

from utils import dists


class Generator:
    """Generate federated learning training and testing data."""

    def __init__(self):
        self.labels = None
        self.trainset = None
        self.trainset_size = None
        self.shards = None


    def read(self, path):
        """Abstract function to read the dataset (should never be called)."""
        raise NotImplementedError


    def group(self):
        """Group the data by label."""
        # Create empty dict of labels
        grouped_data = {label: []
                        for label in self.labels}

        # Populate the grouped data dict
        for datapoint in self.trainset:
            __, label = datapoint  # Extract the label
            label = self.labels[label]

            grouped_data[label].append(datapoint)

        self.trainset = grouped_data  # Overwrite the trainset with grouped data by label


    def generate(self, path):
        """Run the data generation process."""
        self.read(path)
        self.trainset_size = len(self.trainset)  # Extract the trainset size
        self.group()

        return self.trainset


class Loader:
    """Load IID data partitions."""

    def __init__(self, config, generator):
        """Get data from the generator."""
        self.config = config
        self.trainset = generator.trainset
        self.testset = generator.testset
        self.labels = generator.labels
        self.trainset_size = generator.trainset_size

        # Store used data seperately
        self.used = {label: [] for label in self.labels}
        self.used['testset'] = []

    def extract(self, label, n):
        """Extract the data for a particular label."""
        if len(self.trainset[label]) > n:
            extracted = self.trainset[label][:n]  # Extract the data
            self.used[label].extend(extracted)  # Move data to used
            del self.trainset[label][:n]  # Remove from the trainset
            return extracted
        else:
            logging.warning('Insufficient data in label: %s', label)
            logging.warning('Dumping used data for reuse')

            # Unmark data as used
            for label in self.labels:
                self.trainset[label].extend(self.used[label])
                self.used[label] = []

            # Extract replenished data
            return self.extract(label, n)

    def get_partition(self, partition_size):
        """Get a partition that is uniform across all the labels."""
        # Use uniform distribution
        dist = dists.uniform(partition_size, len(self.labels))

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition

    def get_testset(self):
        """Return the entire testset."""
        return self.testset


class BiasLoader(Loader):
    """Load and pass 'preference bias' data partitions."""

    def get_partition(self, partition_size, pref):
        """Get a non-uniform partition with a preference bias."""
        # Extract bias configuration from config
        bias = self.config.data.bias_primary_percentage
        secondary = self.config.data.bias_secondary_focus

        # Calculate sizes of majorty and minority portions
        majority = int(partition_size * bias)
        minority = partition_size - majority

        # Calculate the number of minor labels
        len_minor_labels = len(self.labels) - 1

        if secondary:
            # Distribute to random secondary label
            dist = [0] * len_minor_labels
            dist[random.randint(0, len_minor_labels - 1)] = minority
        else:
            # Distribute among all minority labels
            dist = dists.uniform(minority, len_minor_labels)

        # Add majority data to distribution
        dist.insert(self.labels.index(pref), majority)

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition


class ShardLoader(Loader):
    """
    Load data partitions with sharding, which means data is to be horizontally partitioned (in
    database terminologies).
    """
    def __init__(self):
        self.shards = None

    def create_shards(self):
        """Create all the shards (partitions) from the data."""
        # Extract the number of shards per client from the configuration
        per_client = self.config.data.shard_per_client

        # Determine the correct total number of shards and the size of each shard
        total = self.config.clients.total * per_client
        shard_size = int(self.trainset_size / total)

        data = []
        for __, items in self.trainset.items():
            data.extend(items)

        shards = [data[(i * shard_size):((i + 1) * shard_size)]
                  for i in range(total)]
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
        per_client = self.config.data.shard_per_client

        # Create data partition
        partition = []
        for _ in range(per_client):
            partition.extend(self.extract_shard())

        # Shuffle data partition
        random.shuffle(partition)

        return partition
