'''
Samples data from a dataset, 1). biased across labels, and 2). the number of labels
 (corresponding classes) in different clients is the same, and 3). part of clients
 share same classes 4) the classes shared by these clients are partly used by other
 clients.

This sampler implements the basic label quantity noniid as that in "label_quantity_noniid.py".
 However, part of clients "consistent_clients_size" contain same classes "anchor_classes". Then,
 the "keep_anchor_classes_size" classes of "consistent_clients" are also used in classes pool
 to complete the class assignment.

    For Example:
        Setting per_client_classes_size = 3, anchor_classes=[2, 3, 9], consistent_clients=[0,1,N],
         keep_anchor_classes_size=1 will induce the condition:

                classes 1       2       3 ...     7       8     9
                client1 0      350      350       0       0     350
                client2 0      350      350       0       0     350
                client3 100     20      0         0      100    0
                client4 100     0       0         100    100    0
                ...
                clientN 0       350     350     0       0      350

        We have N clients while K clients contain class c. As class c contains D_c samples,
         each client in K will contain D_c / K samples of this class.
'''

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

from plato.config import Config
from plato.samplers import base

from plato.samplers import sampler_utils


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across classes according to the parameter per_client_classes_size."""
    def __init__(self, datasource, client_id, testing):
        super().__init__()
        self.client_id = client_id

        # Different clients should share the randomness
        #  as the assignment of classes is completed in each
        #  sampling process.
        # Thus, they share the clients_dataidx_map
        np.random.seed(self.random_seed)

        per_client_classes_size = Config().data.per_client_classes_size
        anchor_classes = Config().data.anchor_classes
        consistent_clients_size = Config().data.consistent_clients_size
        keep_anchor_classes_size = Config().data.keep_anchor_classes_size
        total_clients = Config().clients.total_clients

        assert per_client_classes_size == len(anchor_classes)

        self.consistent_clients = np.random.choice(
            list(range(total_clients)),
            size=consistent_clients_size,
            replace=False)
        self.anchor_classes = anchor_classes
        self.keep_anchor_classes_size = keep_anchor_classes_size

        # obtain the dataset information
        if testing:
            target_list = datasource.get_test_set().targets
        else:
            # the list of labels (targets) for all the examples
            target_list = datasource.get_train_set().targets

        self.targets_list = target_list
        classes_text_list = datasource.classes()
        classes_id_list = list(range(len(classes_text_list)))

        self.clients_dataidx_map = {
            client_id: np.ndarray(0, dtype=np.int64)
            for client_id in range(total_clients)
        }
        # construct the quantity label skewness
        self.quantity_label_skew(
            dataset_labels=self.targets_list,
            dataset_classes=classes_id_list,
            num_clients=total_clients,
            per_client_classes_size=per_client_classes_size)

        self.subset_indices = self.clients_dataidx_map[client_id]

    def quantity_label_skew(self, dataset_labels, dataset_classes, num_clients,
                            per_client_classes_size):
        """ Achieve the quantity-based lable skewness """
        client_id = self.client_id
        # each client contains the full classes
        if per_client_classes_size == len(dataset_classes):
            self.clients_dataidx_map = sampler_utils.assign_fully_classes(
                dataset_labels, dataset_classes, num_clients, client_id)
        else:
            self.clients_dataidx_map = sampler_utils.assign_sub_classes(
                dataset_labels,
                dataset_classes,
                num_clients,
                per_client_classes_size,
                anchor_classes=self.anchor_classes,
                consistent_clients=self.consistent_clients,
                keep_anchor_classes_size=self.keep_anchor_classes_size)

    def get(self):
        """Obtains an instance of the sampler. """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)

        return SubsetRandomSampler(self.subset_indices, generator=gen)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return len(self.subset_indices)

    def get_trainset_condition(self):
        """ Obtain the detailed information in the trainser """
        targets_array = np.array(self.targets_list)
        client_sampled_subset_labels = targets_array[self.subset_indices]
        unique, counts = np.unique(client_sampled_subset_labels,
                                   return_counts=True)
        return np.asarray((unique, counts)).T
