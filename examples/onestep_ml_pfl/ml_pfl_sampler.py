'''
Label Distribution Skew:

This sampler is one type of label distribution skew, that is:

    Quantity-based label imbalance: each party owns data samples of a fixed number of labels.

For one client, it contain the defined number of classes while samples in each
class are almost the same. The main reason is that all samples for one class are
randomly assigned to clients that contain this class.
'''

import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

from plato.config import Config
from plato.samplers import base


def generate_left_classes_pool(anchor_classes,
                               all_classes,
                               keep_anchor_size=1):

    left_anchor_classes = np.random.choice(anchor_classes,
                                           size=keep_anchor_size,
                                           replace=False)
    left_classes_id_list = [
        class_id for class_id in all_classes if class_id not in anchor_classes
    ]
    left_classes_id_list = left_anchor_classes.tolist() + left_classes_id_list

    return left_classes_id_list


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a divided partition of the
    dataset, biased across labels according to the Dirichlet distribution."""
    def __init__(self, datasource, client_id, sampler_type="train"):
        super().__init__()
        self.client_id = client_id

        total_clients = Config().clients.total_clients
        consistent_clients_size = Config().data.consistent_clients_size
        # select clients with the same classes
        # using the same random seed
        np.random.seed(self.random_seed)
        self.consistent_clients = np.random.choice(
            list(range(total_clients)),
            size=consistent_clients_size,
            replace=False)

        # Different clients should have a different bias across the labels
        np.random.seed(self.random_seed * int(client_id))

        per_client_classes_size = Config().data.per_client_classes_size

        # The list of labels (targets) for all the examples
        if sampler_type == "train":
            self.targets_list = datasource.targets()
        else:
            self.targets_list = datasource.test_targets()

        classes_text_list = datasource.classes()
        classes_id_list = list(range(len(classes_text_list)))

        # we assign consistent 4 classes for the selected clients
        self.anchor_classes = [0, 1, 2, 3]
        self.maintained_anchor_classes_size = 1

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

        # we need to decrease the client id here
        client_id = client_id - 1
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

            if client_id in self.consistent_clients:
                current_assigned_cls = self.anchor_classes
                for assigned_cls in current_assigned_cls:
                    classes_assigned_count[assigned_cls] += 1
            else:
                left_classes_id_list = generate_left_classes_pool(
                    anchor_classes=self.anchor_classes,
                    all_classes=dataset_classes,
                    keep_anchor_size=1)

                num_classes = len(left_classes_id_list)
                current_assigned_cls_idx = client_id % num_classes
                assigned_cls = dataset_classes[current_assigned_cls_idx]
                current_assigned_cls = [assigned_cls]
                classes_assigned_count[assigned_cls] += 1
                j = 1
                while j < per_client_classes_size:
                    # ind = random.randint(0, max_class_id - 1)
                    ind = np.random.choice(left_classes_id_list, size=1)[0]
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

    def obtain_selected_consistent_clients(self):
        """ Obtain the consistent clients id """
        return self.consistent_clients