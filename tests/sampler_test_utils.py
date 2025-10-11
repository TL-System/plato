"""
Useful functions to test the correctness of samplers

"""

import logging
from collections import Counter, OrderedDict

import numpy as np
import torch


def dicts_equal(dict1, dict2):
    """Whether two dicts are identical."""
    return dict1.keys() == dict2.keys() and dict1.values() == dict2.values()


def align_lists(collected_data_list):
    """Cut the list elem in collected_data_list by using its minimum list length."""
    min_size = np.min([len(elem) for elem in collected_data_list])
    aligned_list = [elem[:min_size] for elem in collected_data_list]

    return aligned_list


def extract_equality_info(collected_data_info):
    """Extract the keys that have a similar value by using matrix operation.

    Returns:
        non_equality_count (array): 1d array, each elem presents how many
                                    clients are different from current
                                    correponding  client
    """

    if isinstance(collected_data_info, list):
        collected_data_info = align_lists(collected_data_info)
        collected_data_info = np.array(collected_data_info)

    # Converting to 2D
    if collected_data_info.ndim == 1:
        collected_data_info = np.reshape(collected_data_info, (1, -1))

    init_values_a = np.expand_dims(collected_data_info, axis=0)
    init_values_b = np.expand_dims(collected_data_info, axis=1)

    equalit_compute = init_values_a - init_values_b
    aggrated_equalit_compute = np.sum(equalit_compute, axis=2)

    non_equality_count = np.count_nonzero(aggrated_equalit_compute, axis=1)

    return non_equality_count


def define_sampler(Sampler, dataset_source, client_id, is_testing=False):
    """Define the sampler based on the received argument."""
    if isinstance(Sampler, type):
        defined_sampler = Sampler(
            datasource=dataset_source, client_id=client_id, testing=is_testing
        )
    else:
        defined_sampler = Sampler.get(
            datasource=dataset_source, client_id=client_id, testing=is_testing
        )
    return defined_sampler


def get_phase_dataset(dataset_source, is_test_phase):
    """Obtain the dataset of the required phase train/test."""
    if not is_test_phase:
        dataset = dataset_source.get_train_set()
    else:
        dataset = dataset_source.get_test_set()

    return dataset


def extract_assigend_data_info(dataset, sampler, num_of_batches=None, batch_size=5):
    """Obtain the data information including the classes
    contained and the samples assigned to each class.

    Returns:
        data_size (int): size of the dataset
        assigned_classes_info (ordereddict): classes information of data,
                                    i.e., class_id: sample_size
    """

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=False, batch_size=batch_size, sampler=sampler.get()
    )
    if num_of_batches is None:
        num_of_batches = len(data_loader)

    # record all the classes that are assigned to this client
    assigned_samples_labels = list()
    # record information of classes assigned to this client
    assigned_classes_info = list()

    extract_idx = 0
    for _, labels in data_loader:
        batch_labels = labels.tolist()
        assigned_samples_labels += batch_labels

        extract_idx += 1
        if extract_idx > num_of_batches:
            break

    # obtain the class: sample_size informarion
    cls_sample_info = dict(Counter(assigned_samples_labels))
    assigned_classes_info = OrderedDict(sorted(cls_sample_info.items()))
    data_size = len(assigned_samples_labels)
    return data_size, assigned_classes_info


def extract_assigend_sample_indexs_info(
    dataset, sampler, num_of_batches=None, batch_size=5
):
    """Extract the samples index that are sampled."""
    samples_index = list(range(len(dataset)))
    data_loader = torch.utils.data.DataLoader(
        dataset=samples_index,
        shuffle=False,
        batch_size=batch_size,
        sampler=sampler.get(),
    )
    if num_of_batches is None:
        num_of_batches = len(data_loader)

    # record the obtained samples index
    assigend_samples_index = list()
    extract_idx = 0
    for sample_index in data_loader:
        sample_index = sample_index.numpy().tolist()
        assigend_samples_index.extend(sample_index)

        extract_idx += 1
        if extract_idx > num_of_batches:
            break

    return assigend_samples_index


def collect_clients_data_info(
    clients_id,
    Sampler,
    dataset_source,
    num_of_batches=None,
    batch_size=5,
    is_test_phase=False,
    is_presented=False,
):
    """Collect the required 'clients_id' data information.

    Outputs:
        clients_samples_info [dict]: Contains client_id: [assigned samples index]
        clients_classes_info [dict]: Contains client_id: [assigned classes]
        clients_classes_sample_info [dict]: Contains client_id: {class_id: sample_size}]
        clients_global_info [dict]: Contains the global information
                                    classes_number: [assigned classes of required classes]
                                    samples_number: [local data size of required classes]

        For example:
            clients_classes_info:  {1: orderedDict(class_id: sample_size)}
            clients_samples_info:  {1: [assigned sample index in original dataset]}
            clients_global_info:  {'classes_number': [5, 5, 5, 5, 5],
                                'samples_number': [1016, 1016, 1016, 1016, 1016]}

    """
    assert isinstance(clients_id, list)

    dataset = get_phase_dataset(dataset_source, is_test_phase)

    clients_samples_info = dict()
    clients_classes_info = dict()

    clients_classes_info = dict()

    clients_global_info = dict()
    clients_global_info["classes_number"] = list()
    clients_global_info["samples_number"] = list()

    for client_id in clients_id:
        defined_sampler = define_sampler(Sampler, dataset_source, client_id)

        client_sample_size, client_classes_info = extract_assigend_data_info(
            dataset,
            sampler=defined_sampler,
            num_of_batches=num_of_batches,
            batch_size=batch_size,
        )

        assigend_samples_index = extract_assigend_sample_indexs_info(
            dataset,
            sampler=defined_sampler,
            num_of_batches=num_of_batches,
            batch_size=batch_size,
        )

        client_classes = list(client_classes_info.keys())
        clients_samples_info[client_id] = assigend_samples_index
        clients_classes_info[client_id] = client_classes_info
        clients_global_info["classes_number"].append(len(client_classes))
        clients_global_info["samples_number"].append(client_sample_size)

        if is_presented:
            logging.info("Client: %s", client_id)
            logging.info("Client's total samples: %s", client_sample_size)
            logging.info("Client's classes: %s", " ".join(map(str, client_classes)))
            logging.info(
                "Client's classes sample: %s", " ".join(map(str, client_classes_info))
            )

    if is_presented:
        logging.info(
            "Clients' classes sample: {}".format(
                " ".join(map(str, clients_global_info["classes_number"]))
            )
        )

        logging.info(
            "Clients' samples size: {}".format(
                " ".join(map(str, clients_global_info["samples_number"]))
            )
        )

    return clients_classes_info, clients_samples_info, clients_global_info


def verify_working_correctness(
    Sampler,
    dataset_source,
    client_id,
    num_of_batches=10,
    batch_size=5,
    is_test_phase=False,
):
    """Ensure that a provided sampler can work well in a provided dataset."""

    defined_sampler = define_sampler(Sampler, dataset_source, client_id)

    dataset = get_phase_dataset(dataset_source, is_test_phase)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        sampler=defined_sampler.get(),
    )

    extract_idx = 0
    for examples, _ in data_loader:
        examples = examples.view(len(examples), -1)

        extract_idx += 1
        if extract_idx > num_of_batches:
            break


def verify_client_data_correctness(
    Sampler,
    dataset_source,
    client_id,
    num_of_iterations=5,
    batch_size=5,
    is_test_phase=False,
    is_presented=False,
):
    """Verify that the data information generated for this client by the sampler is same
    all the time.

    It mainly verifies:
     1- the assigned classes
     2- the assigned sample size for each class
     3- the samples index assigned to the client
    """

    assert num_of_iterations > 1  # several steps required to ensure the verify
    verify_flag = False
    iter_clients_class_info = list()
    iter_clients_sample_info = list()

    for _ in range(num_of_iterations):
        clients_classes_info, clients_samples_info, _ = collect_clients_data_info(
            clients_id=[client_id],
            Sampler=Sampler,
            dataset_source=dataset_source,
            num_of_batches=None,
            batch_size=batch_size,
            is_test_phase=is_test_phase,
            is_presented=is_presented,
        )
        client_class_info = clients_classes_info[client_id]
        client_sample_info = clients_samples_info[client_id]
        client_classes = list(client_class_info.keys())
        client_classes_sample_size = list(client_class_info.values())
        iter_clients_class_info.append(client_classes + client_classes_sample_size)
        iter_clients_sample_info.append(client_sample_info)

    # Extracting the equality info by using the matrix computation
    classes_diff_clients_count = extract_equality_info(iter_clients_class_info)
    samples_diff_clients_count = extract_equality_info(iter_clients_sample_info)
    is_consistent_classes = classes_diff_clients_count == 0
    is_consistent_samples = samples_diff_clients_count == 0

    if is_consistent_classes.all() and is_consistent_samples.all():
        verify_flag = True

    return verify_flag


def verify_difference_between_clients(
    clients_id,
    Sampler,
    dataset_source,
    num_of_batches=None,
    batch_size=5,
    is_force_class_diff=False,
    is_test_phase=False,
    is_presented=False,
):
    """Check the difference between the clients.

    Args:
        clients_id (list): a list of clients id to be verfied
        Sampler (object): a object used to define a sampler
        ...
        is_force_diff (Boolean): whether to maintain the difference
                                    between clients' by force,
                                Using this para for sample quantity sampler
                                with full client classes
        is_test_phase (Boolean): whether to verify the sampler on
                                    the test dataset
        is_presented (Boolean): whether to present the intermediate results
    """

    verify_flag = True
    clients_classes_info, clients_samples_info, _ = collect_clients_data_info(
        clients_id=clients_id,
        Sampler=Sampler,
        dataset_source=dataset_source,
        num_of_batches=num_of_batches,
        batch_size=batch_size,
        is_test_phase=is_test_phase,
        is_presented=is_presented,
    )

    # sort the dict by the client id from small to big
    clients_classes_info = OrderedDict(sorted(clients_classes_info.items()))
    clients_samples_info = OrderedDict(sorted(clients_samples_info.items()))

    clients_data = [
        list(cli_cls_info.keys()) + list(cli_cls_info.values())
        for _, cli_cls_info in list(clients_classes_info.items())
    ]
    clients_sample_data = [
        cli_sample_info for _, cli_sample_info in list(clients_samples_info.items())
    ]

    classes_diff_count = extract_equality_info(clients_data)
    sample_idxs_diff_count_ = extract_equality_info(clients_sample_data)

    is_consistent_classes = classes_diff_count == 0
    is_consistent_samples = sample_idxs_diff_count_ == 0

    if (
        is_force_class_diff and is_consistent_classes.all()
    ) and is_consistent_samples.all():
        verify_flag = False

    return verify_flag


def verify_clients_fixed_classes(
    clients_id,
    Sampler,
    dataset_source,
    required_classes_size,
    num_of_batches=None,
    batch_size=5,
    is_test_phase=False,
    is_presented=False,
):
    """Check whether the clients are assigned required number of classes"""
    verify_flag = True
    clients_classes_info, _, _ = collect_clients_data_info(
        clients_id=clients_id,
        Sampler=Sampler,
        dataset_source=dataset_source,
        num_of_batches=num_of_batches,
        batch_size=batch_size,
        is_test_phase=is_test_phase,
        is_presented=is_presented,
    )

    for client_id in list(clients_classes_info.keys()):
        client_assigned_classes = list(clients_classes_info[client_id].keys())
        if len(client_assigned_classes) != required_classes_size:
            verify_flag = False

    return verify_flag
