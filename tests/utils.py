"""
Useful functions to test the correcness of samplers

"""

import collections
import logging

import torch


def equal_dics(dict1, dict2):
    """ Whether two dicts are same with each other """
    is_equal = False
    for key in dict1.keys():
        if key in dict2.keys():
            if dict1[key] == dict2[key]:
                is_equal = True
            else:
                is_equal = False
        else:
            is_equal = False

    return is_equal


def define_sampler(Sampler, dataset_source, client_id):
    """ Define the sampler based on the received argument """
    if isinstance(Sampler, type):
        defined_sampler = Sampler(datasource=dataset_source,
                                  client_id=client_id)
    else:
        defined_sampler = Sampler.get(datasource=dataset_source,
                                      client_id=client_id)
    return defined_sampler


def get_phase_dataset(dataset_source, is_test_phase):
    """ Obtain the dataset of the required phase train/test """
    if not is_test_phase:
        dataset = dataset_source.get_train_set()
    else:
        dataset = dataset_source.get_test_set()

    return dataset


def extract_assigend_data_info(dataset,
                               sampler,
                               num_of_batches=None,
                               batch_size=5):
    """ Obtain the data information including the classes
    contained and the samples assigned to each class """

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              shuffle=False,
                                              batch_size=batch_size,
                                              sampler=sampler.get())
    if num_of_batches is None:
        num_of_batches = len(data_loader)
    # record all the classes that are assigned to this client
    assigned_samples_labels = list()
    # record the samples contaned in each classes
    assigned_classes_sample = dict()
    # record what classes are assigned to this client
    assigned_classes = list()

    extract_idx = 0
    for _, labels in data_loader:
        batch_labels = labels.tolist()
        assigned_samples_labels += batch_labels

        extract_idx += 1
        if extract_idx > num_of_batches:
            break

    assigned_classes_sample = dict(
        collections.Counter(assigned_samples_labels))
    assigned_classes = sorted(list(set(assigned_samples_labels)))
    return len(
        assigned_samples_labels), assigned_classes, assigned_classes_sample


def extract_assigend_sample_indexs_info(dataset,
                                        sampler,
                                        num_of_batches=None,
                                        batch_size=5):
    """ Extracting the samples index that are sampled """
    samples_index = list(range(len(dataset)))
    data_loader = torch.utils.data.DataLoader(dataset=samples_index,
                                              shuffle=False,
                                              batch_size=batch_size,
                                              sampler=sampler.get())
    if num_of_batches is None:
        num_of_batches = len(data_loader)

    # record the obtained samples index
    assigend_samples_index = list()
    extract_idx = 0
    for sample_index in data_loader:
        sample_index = sample_index.numpy().tolist()
        assigend_samples_index.append(sample_index)

        extract_idx += 1
        if extract_idx > num_of_batches:
            break

    return assigend_samples_index


def collect_clients_data_info(clients_id,
                              Sampler,
                              dataset_source,
                              num_of_batches=None,
                              batch_size=5,
                              is_test_phase=False,
                              is_presented=False):
    """ Collect the required 'clients_id' data information

    Outputs:
        clients_samples_info [dict]: Contains client_id: [assigned samples index]
        clients_classes_info [dict]: Contains client_id: [assigned classes]
        clients_classes_sample_info [dict]: Contains client_id: {class_id: sample_size}]
        clients_global_info [dict]: Contains the global information
                                    classes_number: [assigned classes of required classes]
                                    samples_number: [local data size of required classes]

        For example:
            clients_classes_info:  {1: [1, 2, 6, 7, 9]}
            clients_classes_sample_info:  {1: {6: 200, 2: 218, 9: 186, 1: 239, 7: 173}}
            clients_global_info:  {'classes_number': [5, 5, 5, 5, 5],
                                'samples_number': [1016, 1016, 1016, 1016, 1016]}

    """
    assert isinstance(clients_id, list)

    dataset = get_phase_dataset(dataset_source, is_test_phase)

    clients_samples_info = dict()
    clients_classes_info = dict()

    clients_classes_info = dict()
    clients_classes_sample_info = dict()

    clients_global_info = dict()
    clients_global_info["classes_number"] = list()
    clients_global_info["samples_number"] = list()

    for client_id in clients_id:
        defined_sampler = define_sampler(Sampler, dataset_source, client_id)

        client_total_samples, assigned_classes, \
                assigned_classes_sample = extract_assigend_data_info(
            dataset,
            sampler=defined_sampler,
            num_of_batches=num_of_batches,
            batch_size=batch_size)

        assigend_samples_index = extract_assigend_sample_indexs_info(
            dataset,
            sampler=defined_sampler,
            num_of_batches=num_of_batches,
            batch_size=batch_size)

        clients_samples_info[client_id] = assigend_samples_index
        clients_classes_info[client_id] = assigned_classes
        clients_classes_sample_info[client_id] = assigned_classes_sample
        clients_global_info["classes_number"].append(len(assigned_classes))
        clients_global_info["samples_number"].append(client_total_samples)

        if is_presented:
            logging.info("Client: %d", client_id)
            logging.info("Client's total samples: %d", client_total_samples)
            logging.info("Client's classes: {}".format(' '.join(
                map(str, assigned_classes))))
            logging.info("Client's classes sample: {}".format(' '.join(
                map(str, assigned_classes_sample))))

    if is_presented:
        logging.info("Clients' classes sample: {}".format(' '.join(
            map(str, clients_global_info["classes_number"]))))

        logging.info("Clients' samples size: {}".format(' '.join(
            map(str, clients_global_info["samples_number"]))))

    return clients_classes_info, clients_classes_sample_info, \
        clients_samples_info, clients_global_info


def verify_working_correcness(Sampler,
                              dataset_source,
                              client_id,
                              num_of_batches=10,
                              batch_size=5,
                              is_test_phase=False):
    """ Ensure the sampler can work well in this dataset """

    defined_sampler = define_sampler(Sampler, dataset_source, client_id)

    dataset = get_phase_dataset(dataset_source, is_test_phase)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              shuffle=False,
                                              batch_size=batch_size,
                                              sampler=defined_sampler.get())

    extract_idx = 0
    for examples, _ in data_loader:

        examples = examples.view(len(examples), -1)

        extract_idx += 1
        if extract_idx > num_of_batches:
            break


def verify_client_data_correcness(Sampler,
                                  dataset_source,
                                  client_id,
                                  num_of_iterations=5,
                                  batch_size=5,
                                  is_test_phase=False,
                                  is_presented=False):
    """ Verify the data information generated for this client by the sampler is same all the time

        It mainly verify:
         1- the assigned classes
         2- the assigned sample size for each class
         3- the samples index assigned to the client
    """

    assert num_of_iterations > 1  # several steps required to ensure the verify
    verify_flag = False
    verify_count = 0
    client_anchor_classes = None
    client_anchor_classes_sample_size = None
    client_anchor_samples_index = None

    for itera_idx in range(num_of_iterations):

        clients_classes_info, \
            clients_classes_sample_info, clients_samples_info, \
                _ \
                = collect_clients_data_info(clients_id=[client_id],
                                Sampler=Sampler,
                                dataset_source=dataset_source,
                                num_of_batches=None,
                                batch_size=batch_size,
                                is_test_phase=is_test_phase,
                                is_presented=is_presented)
        if itera_idx == 0:  # the first iteration
            client_anchor_classes = clients_classes_info[client_id]
            client_anchor_classes_sample_size = clients_classes_sample_info[
                client_id]
            client_anchor_samples_index = clients_samples_info[client_id]
            verify_count += 1
        else:

            is_same_classes = client_anchor_classes == \
                                clients_classes_info[client_id]
            is_same_cls_sample_size = equal_dics(
                client_anchor_classes_sample_size,
                clients_classes_sample_info[client_id])
            is_same_samples_index = client_anchor_samples_index == \
                                        clients_samples_info[client_id]

            if is_same_classes and is_same_cls_sample_size and is_same_samples_index:
                verify_count += 1

    if verify_count == num_of_iterations:
        verify_flag = True

    return verify_flag


def verify_difference_between_clients(clients_id,
                                      Sampler,
                                      dataset_source,
                                      num_of_batches=None,
                                      batch_size=5,
                                      is_test_phase=False,
                                      is_presented=False):
    """ Check the difference between the clients """

    verify_flag = True
    clients_classes_info, \
        clients_classes_sample_info, \
            _, _ = collect_clients_data_info(clients_id=clients_id,
                              Sampler=Sampler,
                              dataset_source=dataset_source,
                              num_of_batches=num_of_batches,
                              batch_size=batch_size,
                              is_test_phase=is_test_phase,
                              is_presented=is_presented)

    clients = list(clients_classes_info.keys())
    for client_id in clients:
        client_classes = clients_classes_info[client_id]
        client_samples = clients_classes_sample_info[client_id]
        for check_client_id in clients:
            if check_client_id != client_id:
                check_client_classes = clients_classes_info[check_client_id]
                check_client_samples = clients_classes_sample_info[
                    check_client_id]
                if check_client_classes == client_classes \
                    and check_client_samples == client_samples:
                    verify_flag = False

    return verify_flag


def verify_clients_fixed_classes(clients_id,
                                 Sampler,
                                 dataset_source,
                                 required_classes_size,
                                 num_of_batches=None,
                                 batch_size=5,
                                 is_test_phase=False,
                                 is_presented=False):
    """ Check the clients are assigned required number of classes """
    verify_flag = True
    clients_classes_info, \
        _, _, _ = collect_clients_data_info(clients_id=clients_id,
                              Sampler=Sampler,
                              dataset_source=dataset_source,
                              num_of_batches=num_of_batches,
                              batch_size=batch_size,
                              is_test_phase=is_test_phase,
                              is_presented=is_presented)

    for client_id in list(clients_classes_info.keys()):
        client_assigned_classes = clients_classes_info[client_id]
        if len(client_assigned_classes) != required_classes_size:
            verify_flag = False

    return verify_flag
