"""
The implementation of revise one keyword of a list of config files.


"""

import argparse
import os
from typing import Any, IO

import glob

current_path = "./"

methods_root_dir = os.path.join(current_path, "examples",
                                "contrastive_adaptation")

config_files_root_dir = os.path.join(methods_root_dir, "configs")

base_port_id = 8000


def revise_str_config_file_portID(file_root_dir,
                                  file_name,
                                  port_id,
                                  to_save_dir=None):
    """ Revise the file based on the plain text.

        Using this function can maintain the original
        config file structure and comments
    """

    file_path = os.path.join(file_root_dir, file_name)

    text_lines = []

    with open(file_path, 'r', encoding="utf-8") as fl:
        text_lines = fl.readlines()

    revised_text_lines = []
    for txt_line in text_lines:
        if "port: " in txt_line:
            txt_line = "    port: " + str(port_id) + "\n"
        revised_text_lines.append(txt_line)

    if to_save_dir is None:
        to_save_path = os.path.join(file_root_dir, file_name)
    else:
        to_save_path = os.path.join(to_save_dir, file_name)

    with open(to_save_path, 'w', encoding="utf-8") as fl:
        for line in revised_text_lines:
            fl.write(line)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--sentence',
                        type=str,
                        default='running_mode: ',
                        help='the .')

    parser.add_argument('-c',
                        '--change',
                        type=str,
                        default='running_mode: script',
                        help='The key word of desired scripts.')

    args = parser.parse_args()

    key_sentence = args.sentence
    change_to_sentence = args.change

    configs_files_path = glob.glob(
        os.path.join(config_files_root_dir, "*/", "*.yml"))

    to_save_dir = os.path.join(config_files_root_dir, "test_revision")

    for file_path in configs_files_path:
        # obtain the full path of the existed dir of the file
        # such as ./examples/contrastive_adaptation/configs/whole_global_model
        file_dir_path = os.path.dirname(file_path)
        # obtain the config file name
        file_name = os.path.basename(file_path)

        revise_str_config_file_portID(file_dir_path,
                                      file_name,
                                      port_id=base_port_id,
                                      to_save_dir=file_dir_path)
        base_port_id += 1
