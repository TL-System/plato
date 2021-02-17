"""
Load data partitions in a 'mixed' fashion.
Some clients' datasets are iid, some are non-iid (at most have two labels of samples.)
"""

import logging
import random

from dividers import base
from config import Config


class MixedDivider(base.Divider):
    """Load mixed data partitions."""
    def __init__(self, dataset):
        super().__init__(dataset)
        self.sort_trainset()

        self.shards = None
        self.create_shards()

        self.shard_id_dic = {}
        self.get_shard_ids()

    def sort_trainset(self):
        """Sort the training data by label."""
        # Create an empty dictionary of labels
        grouped_data = {label: [] for label in self.labels}

        # Populate the dictionary
        for datapoint in self.trainset:
            __, label = datapoint
            label = self.labels[label]
            grouped_data[label].append(datapoint)

        self.trainset = []
        for __, items in grouped_data.items():
            self.trainset.extend(items)

    def create_shards(self):
        """Create all the shards (partitions) from the data."""

        # Determine the correct total number of shards and the size of each shard
        total_shards = Config().clients.total_clients * len(self.labels)
        shard_size = int(len(self.trainset) / total_shards)

        self.shards = [
            self.trainset[(i * shard_size):((i + 1) * shard_size)]
            for i in range(total_shards)
        ]

        logging.info("Created %s shards of size %s", len(self.shards),
                     shard_size)

    def get_shard_ids(self):
        """Get the shard indexs of all clients."""
        shard_id_dic = {}

        iid_clients_list = []
        non_iid_clients_list = []

        # Extract client id of different distributed datasets from config file
        iid_clients = Config().data.iid_clients
        if isinstance(iid_clients, int):
            # if only one client's dataset is iid
            iid_clients_list = [str(iid_clients)]
        else:
            iid_clients_list = [x.strip() for x in iid_clients.split(',')]

        non_iid_clients = Config().data.non_iid_clients
        if isinstance(non_iid_clients, int):
            # if only one client's dataset is non-iid
            non_iid_clients_list = [str(non_iid_clients)]
        else:
            non_iid_clients_list = [
                x.strip() for x in non_iid_clients.split(',')
            ]

        # The number of shards per client is the number of labels in the whole dataset
        shards_per_client = len(self.labels)

        total_clients = Config().clients.total_clients

        shard_id_list = [i for i in range(shards_per_client * total_clients)]

        # iid clients will have one shard of each label
        for i, client_id in enumerate(iid_clients_list):
            shard_id_dic[client_id] = []
            for j in range(shards_per_client):
                new_shard_id = i + j * total_clients
                shard_id_dic[client_id].append(new_shard_id)
                shard_id_list.remove(new_shard_id)

        for _, client_id in enumerate(non_iid_clients_list):
            shard_id_dic[client_id] = shard_id_list[:shards_per_client]
            del shard_id_list[:shards_per_client]

        self.shard_id_dic = shard_id_dic

    def get_partition(self, client_id):
        """Get a partition for a client."""

        # The index of shard that loaded to this client
        shard_id_list = self.shard_id_dic[client_id]

        # Create data partition
        partition = []
        for shard_id in shard_id_list:
            partition.extend(self.shards[shard_id])

        # Shuffle data partition
        random.shuffle(partition)

        return partition
