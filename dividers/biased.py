import random

from dividers import base
from utils import dists
from config import Config


class BiasedDivider(base.Divider):
    """Load and pass 'preference bias' data partitions."""
    def __init__(self, dataset):
        super().__init__(dataset)
        self.group()
        self.shards = None

    def get_partition(self, partition_size, pref):
        """Get a non-uniform partition with a preference bias."""
        # Extract bias configuration from config
        bias = Config().data.bias_primary_percentage
        secondary = Config().data.bias_secondary_focus

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
            dist, __ = dists.uniform(minority, len_minor_labels)

        # Add majority data to distribution
        dist.insert(self.labels.index(pref), majority)

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition
