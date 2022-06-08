"""
Convert the raw data to be format.

As the personalized accuracies of clients are stored in the .csv file following the
 format:
 round, client_id, accuracy
    xx      xx      xx

I am going to convert it to be expected format such as:
 communication round, client_1, client_2, client_3, ...,
            xx          acc1,       acc2,   acc3

"""

from collections import OrderedDict
import os

import numpy as np
import pandas as pd


def personalized_acc_to_format(file_path):
    """ Convert the generated personalized accuracy csv file to the formet one. """
    round_name = 'round'
    client_prefix = "client_"

    per_acc_dataframe = pd.read_csv(file_path)
    # obtain the number of communication rounds
    rounds_pool = per_acc_dataframe[round_name].unique()
    # obtain the clients
    clients_pool = per_acc_dataframe['client_id'].unique()
    clients_pool = np.sort(clients_pool)

    round_col_name = [round_name]
    clients_col_name = [client_id for client_id in clients_pool]
    round_cols_name = round_col_name + clients_col_name

    def insert_round_clients(base_df, round_group_df):

        cliends_acc_df = round_group_df[['client_id', 'accuracy']]
        cliends_acc_df = cliends_acc_df.sort_values('client_id')
        round_row = {round_name: round_id}
        row_clients_col = dict(
            zip(cliends_acc_df.client_id, cliends_acc_df.accuracy))
        round_row.update(row_clients_col)

        #base_round_row =
        round_row_df = pd.DataFrame(round_row, index=[0])

        base_df = pd.concat([base_df, round_row_df], ignore_index=True)
        return base_df

    grouped_df = per_acc_dataframe.groupby([round_name])

    per_acc_new_df = pd.DataFrame(columns=round_cols_name)

    for round_id, round_group in grouped_df:
        per_acc_new_df = insert_round_clients(per_acc_new_df,
                                              round_group_df=round_group)

    # add prefix to clients' col
    clients_col_new_name = [
        client_prefix + str(client_id) for client_id in clients_pool
    ]
    cols_name = dict(zip(clients_col_name, clients_col_new_name))
    per_acc_new_df.rename(columns=cols_name, inplace=True)

    return per_acc_new_df