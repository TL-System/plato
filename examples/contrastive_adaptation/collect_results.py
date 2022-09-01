"""
Collecting the experimental results to present in the experiments.

"""

import os
import json

import numpy as np

import extract_format_data

result_dir = "results"
#exp_name = "CF10labelfix2"
#exp_name = "CF10noniid03"
#exp_name = "CF100labelfix5"
#exp_name = "CF100noniid03"

exp_name = "STL10labelfix2"

methods_name = [
    'fedrep', 'fedavg', 'script', 'apfl', 'fedbabu', 'lgfedavg', 'fedper',
    'perfedavg', 'scaffold', 'ditto'
]


def obtain_data_info(method_results_dir):
    collected_clients_data_info = {}
    all_results_name = os.listdir(method_results_dir)
    data_file_keywordname = "data_statistics"
    clients_dir_name = [
        result_name for result_name in all_results_name
        if result_name.startswith("client_")
    ]
    # visit the result name of clients
    # the name should be like: client_1
    for result_name in clients_dir_name:
        client_pers_path = os.path.join(method_results_dir, result_name)
        target_data_file = [
            file for file in os.listdir(client_pers_path)
            if data_file_keywordname in file
        ][0]
        target_data_file_path = os.path.join(client_pers_path,
                                             target_data_file)

        with open(target_data_file_path, 'r') as fp:
            client_data_info = json.loads(fp.read())

        train_size = client_data_info['train_size']
        collected_clients_data_info[result_name] = train_size

    return collected_clients_data_info


if __name__ == "__main__":
    target_results_dir = os.path.join(result_dir, exp_name)
    all_result_dirs = os.listdir(target_results_dir)
    for method_name in methods_name:
        print("\n")
        print(method_name)
        method_result_dir_name = [
            dir_name for dir_name in all_result_dirs
            if dir_name.startswith(method_name)
        ]
        if not method_result_dir_name:
            print("Skipping")
            continue
        method_result_dir_name = method_result_dir_name[0]
        method_results_dir_path = os.path.join(target_results_dir,
                                               method_result_dir_name)
        print(method_results_dir_path)

        collected_clients_data_info = obtain_data_info(method_results_dir_path)
        clients_train_size = np.array(
            list(collected_clients_data_info.values()))
        print("train size,  mean: %d, var: %f, std: %f " %
              (np.mean(clients_train_size), np.var(clients_train_size),
               np.std(clients_train_size)))

        # obtain the results without performing personalization
        pers_results = extract_format_data.obtain_format_personalized_data(
            method_results_dir_path)
        final_round_epochs_accs = pers_results['rounds_epochs_clients_acc'][-1]
        epochs_id = pers_results['epochs_id']

        for epoch_idx in range(len(final_round_epochs_accs)):
            epoch = epochs_id[epoch_idx]
            epoch_accs = final_round_epochs_accs[epoch_idx]
            final_accs = epoch_accs
            #print(final_accs)
            print("total %d clients, epoch %d, mean: %f, var: %f, std: %f" %
                  (len(final_accs), epoch, np.mean(final_accs),
                   np.var(final_accs), np.std(final_accs)))
