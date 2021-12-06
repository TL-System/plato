'''
Samples data from a dataset, biased across labels, and the number of labels
(corresponding classes) in different clients is the same.

This sampler implements one type of label distribution skew called:

    Quantity-based label imbalance:
        Each client contains a fixed number of classes parameterized by the
         "per_client_classes_size", while the number of samples in each class
         is almost the same. Besides, the classes assigned to each client are
         randomly selected from all classes.

        The samples of one class are equally divided and assigned to clients
         who contain this class. Thus, the samples of different clients
         are mutual-exclusive.

    For Example:
        Setting per_client_classes_size = 2 will induce the condition that each client
         only contains two classes.
                classes 1       2       3 ...       8     9
                client1 100     0       100         0     0
                client2 0      108      100         0     0
                ...
                clientN 0       0       0           100   100

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
        total_clients = Config().clients.total_clients

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
                anchor_classes=None,
                consistent_clients=None,
                keep_anchor_classes_size=None)

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
