"""
Implementation of generating unique port id for all config files.

The port id will start from 8000.


Reuse part of code from Plato's config.

"""

import os
import json
from typing import Any, IO

import glob
import yaml

current_path = "./"

methods_root_dir = os.path.join(current_path, "examples",
                                "contrastive_adaptation")

config_files_root_dir = os.path.join(methods_root_dir, "configs")

base_port_id = 8000


class Loader(yaml.SafeLoader):
    """ YAML Loader with `!include` constructor. """

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self.root_path = os.path.split(stream.name)[0]
        except AttributeError:
            self.root_path = os.path.curdir

        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(
        os.path.join(loader.root_path, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r', encoding='utf-8') as config_file:
        if extension in ('yaml', 'yml'):
            return yaml.load(config_file, Loader)
        elif extension in ('json', ):
            return json.load(config_file)
        else:
            return ''.join(config_file.readlines())


def revise_yml_config_file_portID(file_root_dir,
                                  file_name,
                                  port_id,
                                  to_save_dir=None):
    """ Revise the config file based on the yml structure. """
    yaml.add_constructor('!include', construct_include, Loader)

    file_path = os.path.join(file_root_dir, file_name)

    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding="utf-8") as config_file:
            config = yaml.load(config_file, Loader)

    # change the port id to the desired one
    config['server']['port_id'] = port_id

    if to_save_dir is None:
        to_save_path = os.path.join(file_root_dir, file_name)
    else:
        to_save_path = os.path.join(to_save_dir, file_name)

    os.makedirs(to_save_dir, exist_ok=True)

    with open(to_save_path, 'w', encoding="utf-8") as file:
        _ = yaml.dump(config, file)


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