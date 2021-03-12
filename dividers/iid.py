"""
Load data partitions in an independent and identically distributed fashion.
"""
from dividers import base
from utils import dists


class IIDDivider(base.Divider):
    def __init__(self, datasource):
        super().__init__(datasource)
        self.labels = datasource.classes()
        self.group()

    """Load IID data partitions."""

    def get_partition(self, partition_size):
        """Get a partition that is uniform across all the labels."""
        size_to_extract = partition_size
        partition = []

        while size_to_extract > 0:
            dist, __ = dists.uniform(size_to_extract, len(self.labels))
            examples_extracted = 0

            # Extracting data according to uniform distribution
            for i, label in enumerate(self.labels):
                extracted = self.extract(label, dist[i])
                examples_extracted += len(extracted)
                partition.extend(extracted)

            size_to_extract -= examples_extracted

        return partition

    def get_testset(self):
        """Return the entire testset."""
        return self.testset
