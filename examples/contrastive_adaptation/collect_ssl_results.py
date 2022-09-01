"""
Collecting the experimental results to present in the experiments.

"""

import os
import json

import numpy as np

import extract_format_data

result_dir = "results"
exp_name = "CF10labelfix2"
#exp_name = "CF10noniid03"
# exp_name = "CF100labelfix5"
#exp_name = "CF100noniid03"

ssl_methods = ['byol', 'simclr', 'simsiam', 'mocov2']

if __name__ == "__main__":
    target_results_dir = os.path.join(result_dir, exp_name)
    all_result_dirs = os.listdir(target_results_dir)
    for method_name in ssl_methods:
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

        # obtain the results without performing personalization
        core_results = extract_format_data.obtain_format_core_data(
            method_results_dir_path)
        final_accs = core_results['rounds_clients_acc'][-1]

        print("total %d clients, mean: %f, var: %f, std: %f" %
              (len(final_accs), np.mean(final_accs), np.var(final_accs),
               np.std(final_accs)))
