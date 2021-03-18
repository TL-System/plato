"""
Load data partitions in an independent and identically distributed fashion.
"""
from config import Config


class IIDDivider:
    """Load IID data partitions."""
    def __init__(self, datasource):
        """Get data from the data source."""
        self.datasource = datasource
        self.partition = None

    def get_partition(self, partition_size, client_id):
        """Get a partition that is uniform across all the labels."""
        num_shards = int(self.datasource.num_train_examples() / partition_size)
        shard_id = int(client_id) - 1
        assert num_shards == Config().clients.total_clients

        self.partition = self.datasource.get_train_partition(
            num_shards, shard_id)

        return self.partition

    def trainset_size(self):
        """Return the number of batches in the training data partition."""
        return self.partition.get_dataset_size()
