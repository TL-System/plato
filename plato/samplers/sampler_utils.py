"""
Useful tools used for implementing samplers

"""

import numpy as np


def extend_indices(indices, required_total_size):
    """ Extend the indices to obtain the required total size
         by duplicating the indices """
    # add extra samples to make it evenly divisible, if needed
    if len(indices) < required_total_size:
        while len(indices) < required_total_size:
            indices += indices[:(required_total_size - len(indices))]
    else:
        indices = indices[:required_total_size]
    assert len(indices) == required_total_size

    return indices


def generate_left_classes_pool(anchor_classes,
                               all_classes,
                               keep_anchor_size=1):
    """ Generate classes pool by 1. removng anchor classes from the all classes
        2. randomly select 'keep_anchor_size' from anchor classes to the left
        class pool. """

    if anchor_classes is None:
        return all_classes

    # obtain subset classes from the anchor class
    left_anchor_classes = np.random.choice(anchor_classes,
                                           size=keep_anchor_size,
                                           replace=False)
    # remove the anchor classes from the whole classes
    left_classes_id_list = [
        class_id for class_id in all_classes if class_id not in anchor_classes
    ]

    # combine the left anchor classes and the left whole classes to
    #  obtain the left classes pool for global classes assignmenr
    left_classes_id_list = left_anchor_classes.tolist() + left_classes_id_list

    return left_classes_id_list


def assign_fully_classes(dataset_labels, dataset_classes, num_clients,
                         client_id):
    """ Assign full classes to each client """

    # define the client_id to sample index mapper
    clients_dataidx_map = {
        client_id: np.ndarray(0, dtype=np.int64)
        for client_id in range(num_clients)
    }

    dataset_labels = np.array(dataset_labels)

    for class_id in dataset_classes:
        idx_k = np.where(dataset_labels == class_id)[0]

        # the samples of each class is evenly assigned to this client
        split = np.array_split(idx_k, num_clients)
        clients_dataidx_map[client_id] = np.append(
            clients_dataidx_map[client_id], split[client_id])
    return clients_dataidx_map


def assign_sub_classes(dataset_labels,
                       dataset_classes,
                       num_clients,
                       per_client_classes_size,
                       anchor_classes=None,
                       consistent_clients=None,
                       keep_anchor_classes_size=None):
    """ Assign subset of classes to each client and assign corresponding samples of classes

        Args:
            dataset_labels (list): a list of lables of global samples
            dataset_classes (list): a list containing classes of the dataset
            num_clients (int): total number of clients for classes assignment
            per_client_classes_size (int): the number of classes assigned to each client
            anchor_classes (list, default []): subset of classes assigned to "consistent_clients"
            consistent_clients (list, default []): subset of classes containing same classes
            keep_anchor_classes_size (list, default None): how many classes in anchor are utilized
                                                        in the class pool for global classes
                                                        assignment.
    """
    # define the client_id to sample index mapper
    clients_dataidx_map = {
        client_id: np.ndarray(0, dtype=np.int64)
        for client_id in range(num_clients)
    }
    dataset_labels = np.array(dataset_labels)

    classes_assigned_count = {cls_i: 0 for cls_i in dataset_classes}
    clients_contain_classes = {cli_i: [] for cli_i in range(num_clients)}

    for client_id in range(num_clients):

        if consistent_clients is not None and client_id in consistent_clients:
            current_assigned_cls = anchor_classes
            for assigned_cls in current_assigned_cls:
                classes_assigned_count[assigned_cls] += 1
        else:
            left_classes_id_list = generate_left_classes_pool(
                anchor_classes=anchor_classes,
                all_classes=dataset_classes,
                keep_anchor_size=keep_anchor_classes_size)

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
                clients_dataidx_map[client_id] = np.append(
                    clients_dataidx_map[client_id], split[ids])
                ids += 1
    return clients_dataidx_map


def create_dirichlet_skew(
        total_size,  # the totoal size to generate partitions
        concentration,  # the beta of the dirichlet dictribution
        number_partitions,  # number of partitions
        min_partition_size=None,  # minimum required size for partitions
        is_extend_total_size=False):
    """ Create the distribution skewness based on the dirichlet distribution

        Note:
            is_extend_total_size (boolean) determines whether to generate the
             partitions satisfying min_partition_size by directly extending
             the total data size.
    """
    if min_partition_size is not None:
        if not is_extend_total_size:
            min_size = 0
            while min_size < min_partition_size:
                proportions = np.random.dirichlet(
                    np.repeat(concentration, number_partitions))

                proportions = proportions / proportions.sum()
                min_size = np.min(proportions * total_size)

        else:  # extend the total size to satisfy the minimum requirement
            minimum_proportion_bound = float(min_partition_size / total_size)

            proportions = np.random.dirichlet(
                np.repeat(concentration, number_partitions))

            proportions = proportions / proportions.sum()

            # set the proportion to satisfy the minimum size
            def set_min_bound(proportion):
                if proportion > minimum_proportion_bound:
                    return proportion
                else:
                    return minimum_proportion_bound

            proportions = list(map(set_min_bound, proportions))

    else:
        proportions = np.random.dirichlet(
            np.repeat(concentration, number_partitions))

    return proportions
