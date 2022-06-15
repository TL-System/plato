"""
Implementation of running multiples experiments with Slurm.


The type of the experiments should be provided by:
    - key

Thus, the scripts containing the key will be performed.

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

    experiment_script_files_name = glob.glob(
        os.path.join(script_files_dir, "*.sh"))

    parser = argparse.ArgumentParser()
    parser.add_argument('-k',
                        '--key',
                        type=str,
                        default='all',
                        help='The key word of desired scripts.')

    args = parser.parse_args()

    key_word = args.key

    desired_files = [
        file_name for file_name in experiment_script_files_name
        if is_desired_file(key_word, file_name)
    ]
    for file in desired_files:

        script_file_path = os.path.join(script_files_dir, file)
        print(f"Running script: {script_file_path}")
        os.system("sbatch %s" % script_file_path)
