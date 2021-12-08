"""
Useful functions to test the correctness of samplers


"""
import torch
import collections


def dicts_equal(dict1, dict2):
    """ Whether two dicts are same with each other """
    return dict1.keys() == dict2.keys() and dict1.values() == dict2.values()


def verify_working_correctness(Sampler,
                               dataset_source,
                               client_id,
                               num_of_batches=10,
                               batch_size=5):
    """ Ensure the sampler can work well in this dataset """
    test_sampler = Sampler(datasource=dataset_source, client_id=client_id)
    dataset = dataset_source.get_train_set()

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              shuffle=False,
                                              batch_size=batch_size,
                                              sampler=test_sampler.get())

    extract_idx = 0
    for examples, labels in data_loader:

        examples = examples.view(len(examples), -1)

        print("labels: ", labels)

        extract_idx += 1
        if extract_idx > num_of_batches:
            break


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
    assigned_classes = set(assigned_samples_labels)
    return len(
        assigned_samples_labels), assigned_classes, assigned_classes_sample


def verify_client_local_data_correctness(Sampler,
                                         dataset_source,
                                         client_id,
                                         num_of_iterations=5,
                                         batch_size=5,
                                         is_presented=False):
    """ Verify the local data generated for this client by the sampler is same all the time """

    dataset = dataset_source.get_train_set()

    assert num_of_iterations > 1  # several steps required to ensure the verify
    verify_flag = False
    verify_count = 0
    pervios_iter_classes = list()
    pervios_iter_classes_sample = list()
    for iter_i in range(num_of_iterations):
        # we redefine the sampler in each iteration
        test_sampler = Sampler(datasource=dataset_source, client_id=client_id)

        total_samples_n, assigned_classes, assigned_classes_sample = extract_assigend_data_info(
            dataset, test_sampler, num_of_batches=None, batch_size=batch_size)

        if is_presented:
            print("The iteration: ", iter_i)
            print("assigned_classes: ", assigned_classes)
            print("assigned_classes_sample: ", assigned_classes_sample)
            print(("Total samples: {}").format(total_samples_n))

        # this first iteration, empty
        if not pervios_iter_classes and not pervios_iter_classes_sample:
            pervios_iter_classes = assigned_classes
            pervios_iter_classes_sample = assigned_classes_sample
            verify_count += 1

            continue

        if pervios_iter_classes == assigned_classes and dicts_equal(
                pervios_iter_classes_sample, assigned_classes_sample):
            verify_count += 1

        if is_presented:
            print("verify_count: ", verify_count)

    if verify_count == num_of_iterations:
        verify_flag = True

    return verify_flag


def verify_difference_between_clients(clients_id,
                                      Sampler,
                                      dataset_source,
                                      num_of_batches=None,
                                      batch_size=5,
                                      is_presented=False):
    """ Check the difference between the clients """
    assert isinstance(clients_id, list)
    dataset = dataset_source.get_train_set()

    clients_global_info = dict()
    clients_global_info["classes_number"] = list()
    clients_global_info["samples_number"] = list()

    for client_id in clients_id:
        test_sampler = Sampler(datasource=dataset_source, client_id=client_id)
        client_total_samples, \
            assigned_classes, \
                assigned_classes_sample = extract_assigend_data_info(
            dataset,
            sampler=test_sampler,
            num_of_batches=num_of_batches,
            batch_size=batch_size)

        clients_global_info["classes_number"].append(len(assigned_classes))
        clients_global_info["samples_number"].append(client_total_samples)
        if is_presented:
            print("Client: ", client_id)
            print("client_total_samples: ", client_total_samples)
            print("assigned_classes: ", assigned_classes)
            print("assigned_classes_sample: ", assigned_classes_sample)

    if is_presented:
        print("clients global info: ")
        print(clients_global_info)
