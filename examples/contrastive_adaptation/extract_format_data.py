"""
The implementation of extracting format data from the dir

"""

import os
import glob
import pandas as pd

from libs.utils import data_to_format


def obtain_format_core_data(dir_path):
    """ Obtain the format data from the dir of clients.
        The core data is the clients' accuracy collected by the server.

        Args:
            dir_path (str): the path of the dir where the data is stored.

    """

    # 1. get the core csv file
    core_csv_file_path = glob.glob(dir_path + '/' + '*_accuracy.csv')[0]
    obtained_core_acc_df = data_to_format.clients_acc_to_format(
        file_path=core_csv_file_path, do_filter_zero=True)

    rounds_clients_acc, rounds_avg_acc, rounds_id = data_to_format.decode_data(
        data_df=obtained_core_acc_df)

    core_data_container = {
        "rounds_clients_acc": rounds_clients_acc,
        "rounds_avg_acc": rounds_avg_acc,
        "rounds_id": rounds_id
    }

    return core_data_container


def obtain_format_personalized_data(dir_path):
    """ Obtain the format data from the dir of clients.

        Args:
            dir_path (str): the path of the dir where the data is stored.
                The data under this dir should include .csv files and multiple
                dirs with name "cliant_"

        outputs:
            clients_data_container (dict): a dict contains
                - rounds_id: contains the id for rounds, such as 10, 20, ..., 150
                - epochs_id: contains the id for epoch, such as 0, 1
                - rounds_epochs_clients_acc: each itme is a tuple containing
                    len(epochs_id) number of arrays A. j-th array where j \in epochs_id, is the
                    clients' personalized accuracies in epoch epochs_id[j]
                - rounds_epochs_clients_id: same structure as the rounds_epochs_clients_acc but
                    each itme contains the clients' id

    """

    # 2. get subdirs
    clients_dir_name = [
        each_dir_name for each_dir_name in os.listdir(dir_path)
        if "client_" in each_dir_name
    ]

    clients_files_path = []
    for dir_name in clients_dir_name:
        client_dir_path = os.path.join(dir_path, dir_name)

        personalized_file_path = glob.glob(client_dir_path + '/' +
                                           '*_personalization.csv')[0]

        clients_files_path.append(personalized_file_path)

    # obtain the df for clients' per acc
    # columns: round, epoch, client_1_accuracy, client_2_accuracy, ..., client_100_accuracy
    merged_clients_per_acc_df = data_to_format.merge_clients_personalized_acc_to_format(
        clients_files_path)

    rounds_id, epochs_id, rounds_epochs_clients_acc, rounds_epochs_clients_id = data_to_format.decode_clients_per_data(
        data_df=merged_clients_per_acc_df)

    clients_data_container = {
        "rounds_id": rounds_id,
        "epochs_id": epochs_id,
        "rounds_epochs_clients_acc": rounds_epochs_clients_acc,
        "rounds_epochs_clients_id": rounds_epochs_clients_id
    }

    return clients_data_container


def obtain_format_personalized_encoded_data(dir_path):
    """ Obtain the format encoded data from the dir of clients.

        Args:
            dir_path (str): the path of the dir where the encoded data is stored.
                The data under this dir should include .csv files and multiple
                dirs with name "cliant_"

        Outputs:
            merged_clients_encoded_dict (dit): a nested dict
                {"client_25":
                    {round_id: {epoch_id: {"train": xxx, "test": xxx}} },

                ....
                "client_100":
                    {round_id: {epoch_id: {"train": xxx, "test": xxx}} },
                }

                where round_id and epoch_id are int digital numbers.

            For example:
                clients_id = list(merged_clients_encoded_dict.keys())
                print(clients_id)
                client_rounds = list(merged_clients_encoded_dict['client_25'].keys())
                print(client_rounds)
                client_round_epoch = list(
                    merged_clients_encoded_dict['client_25'][50].keys())
                print(client_round_epoch)
                client_round_epoch_splits = list(
                    merged_clients_encoded_dict['client_25'][50][1].keys())
                print(client_round_epoch_splits)
                print(merged_clients_encoded_dict['client_25'][50][1]['train'].shape)

    """

    # 2. get subdirs
    clients_dir_name = [
        each_dir_name for each_dir_name in os.listdir(dir_path)
        if "client_" in each_dir_name
    ]

    clients_encoded_files_path = []
    for dir_name in clients_dir_name:
        client_dir_path = os.path.join(dir_path, dir_name)

        personalized_encoded_files_path = glob.glob(client_dir_path + '/' +
                                                    '*Encoded.npy')

        clients_encoded_files_path.append(personalized_encoded_files_path)

    # obtain the df for clients' encoded acc
    # columns: round, epoch, client_1_accuracy, client_2_accuracy, ..., client_100_accuracy
    merged_clients_encoded_dict = data_to_format.merge_clients_personalized_acc_to_format(
        clients_encoded_data_file_path=clients_encoded_files_path)

    return merged_clients_encoded_dict
