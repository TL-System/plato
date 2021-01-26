"""
Load data partitions in an independent and identically distributed fashion.
"""
from config import Config


class IIDDivider:
    """Load IID data partitions."""
    def __init__(self, dataset):
        """Get data from the dataset."""
        self.dataset = dataset
        self.partition = None

    def get_partition(self, partition_size, client_id):
        """Get a partition that is uniform across all the labels."""
        num_shards = int(self.dataset.num_train_examples() / partition_size)
        shard_id = int(client_id) - 1
        assert num_shards == Config().clients.total_clients

        self.partition = self.dataset.get_train_partition(num_shards, shard_id)
        return self.partition

    def get_testset(self):
        """Return the entire testset."""
        return self.dataset.get_test_set()

    def trainset_size(self):
        """Return the number of batches in the training data partition."""
        return self.partition.get_dataset_size()
