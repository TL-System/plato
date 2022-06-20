"""
Implementation of running multiples experiments with Slurm.


The type of the experiments should be provided by:
    - key

Thus, the scripts containing the key will be performed.


For example,
to submit all methods with configs using the whole model as the
global model based on the MNIST dataset

python examples/contrastive_adaptation/submit_slurm_jobs.py -d whole_global_model -k MNIST


"""

import argparse
import glob
import os

current_path = "./"

methods_root_dir = os.path.join(current_path, "examples",
                                "contrastive_adaptation")

script_files_dir = os.path.join(methods_root_dir, "run_scripts")


def is_desired_file(key_word, file_name):
    """ Whether the file name is the desired file defiend by key. """

    # if key_word is all, all files are desired.
    if key_word == "all":
        return True
    if key_word in file_name:
        return True

    return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dirname',
        type=str,
        default='whole_global_model',
        help='the dir name in which the config files are stored.')

    parser.add_argument('-k',
                        '--key',
                        type=str,
                        default='all',
                        help='The key word of desired scripts.')

    args = parser.parse_args()

    config_dir_name = args.dirname
    key_word = args.key

    experiment_script_files_path = glob.glob(
        os.path.join(script_files_dir, config_dir_name, "*.sh"))

    desired_files_path = [
        file_path for file_path in experiment_script_files_path
        if is_desired_file(key_word, file_path)
    ]
    for script_file_path in desired_files_path:

        print(f"Running script: {script_file_path}")
        os.system("sbatch %s" % script_file_path)
