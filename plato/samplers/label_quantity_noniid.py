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


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across classes according to the parameter per_client_classes_size."""
    def __init__(self, datasource, client_id, testing):
        super().__init__()
        self.client_id = client_id

        # Different clients should have a different bias across the labels
        np.random.seed(self.random_seed * int(client_id))

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
        # each client contains the full classes
        if per_client_classes_size == len(dataset_classes):
            self.fully_classes_assigned(dataset_labels, dataset_classes,
                                        num_clients)
        else:
            self.specific_classes_assigned(dataset_labels, dataset_classes,
                                           num_clients,
                                           per_client_classes_size)

    def fully_classes_assigned(self, dataset_labels, dataset_classes,
                               num_clients):
        """ Assign full classes to each client """
        dataset_labels = np.array(dataset_labels)
        for class_id in dataset_classes:
            idx_k = np.where(dataset_labels == class_id)[0]

            # the samples of each class is evenly assigned to different clients
            split = np.array_split(idx_k, num_clients)
            for client_id in range(num_clients):
                self.clients_dataidx_map[client_id] = np.append(
                    self.clients_dataidx_map[client_id], split[client_id])

    def specific_classes_assigned(self, dataset_labels, dataset_classes,
                                  num_clients, per_client_classes_size):
        """ Assign specific number of classes to each client """
        dataset_labels = np.array(dataset_labels)

        classes_assigned_count = {cls_i: 0 for cls_i in dataset_classes}
        clients_contain_classes = {cli_i: [] for cli_i in range(num_clients)}

        for client_id in range(num_clients):
            num_classes = len(dataset_classes)
            current_assigned_cls_idx = client_id % num_classes
            assigned_cls = dataset_classes[current_assigned_cls_idx]
            current_assigned_cls = [assigned_cls]
            classes_assigned_count[assigned_cls] += 1

            j = 1
            while j < per_client_classes_size:
                # ind = random.randint(0, max_class_id - 1)
                ind = np.random.choice(dataset_classes, size=1)[0]
                if ind not in current_assigned_cls:
                    j = j + 1
                    current_assigned_cls.append(ind)
                    classes_assigned_count[ind] += 1
            clients_contain_classes[client_id] = current_assigned_cls

        for class_id in dataset_classes:
            # skip if this class is never assinged to any clients
            if classes_assigned_count[class_id] == 0:
                continue

            idx_k = np.where(dataset_labels == class_id)[0]

            # the samples of current class are evenly assigned to the corresponding clients
            split = np.array_split(idx_k, classes_assigned_count[class_id])
            ids = 0
            for client_id in range(num_clients):
                if class_id in clients_contain_classes[client_id]:
                    self.clients_dataidx_map[client_id] = np.append(
                        self.clients_dataidx_map[client_id], split[ids])
                    ids += 1

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
