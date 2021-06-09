#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base class for multimodal datasets.
"""

import logging
import os
import sys
import requests

import numpy as np
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

from plato.datasources import base


class MultiModalDataSource(base.DataSource):
    """
    The training or testing dataset that accommodates custom augmentation and transforms.
    """
    def __init__(self):
        super().__init__()

        # data name
        self.data_name = ""

        # the text name of the contained modalities
        self.modality_names = []

        # define the information container for the source data
        self.mm_data_info = {"source_data_path": "", "base_data_dir_path": ""}

        # define the paths for the splited root data - train, test, and val
        self.splits_info = {
            "train": {
                "path": '',
                "split_file": ''
            },
            "test": {
                "path": '',
                "split_file": ''
            },
            "val": {
                "path": '',
                "split_file": ''
            }
        }

    def _data_path_process(
        self,
        data_path,  # the base directory for the data 
        base_data_name=None):  # the directory name of the working data
        base_data_path = os.path.join(data_path, base_data_name)
        if not os.path.exists(base_data_path):
            os.makedirs(base_data_path)

        self.mm_data_info["base_data_dir_path"] = base_data_path

        # create the split dirs for current dataset
        for split_type in list(self.splits_info.keys()):
            split_path = os.path.join(base_data_path, split_type)
            self.splits_info[split_type]["path"] = split_path
            if not os.path.exists(split_path):
                try:
                    os.makedirs(split_path)
                except FileExistsError:
                    pass

    def _download_arrange_data(
        self,
        download_url,
        put_data_dir,
    ):
        download_file_name = download_url.split("/")[-1]
        download_extracted_file_name = download_file_name.split(".")[0]
        download_extracted_dir_path = os.path.join(
            put_data_dir, download_extracted_file_name)
        # download the raw data if necessary
        if not self._exist_judgement(download_extracted_dir_path):
            logging.info(
                ("Downloading the {} data.....").format(download_file_name))
            download_and_extract_archive(url=download_url,
                                         download_root=put_data_dir,
                                         extract_root=put_data_dir,
                                         filename=download_file_name,
                                         remove_finished=True)

    def _download_google_driver_arrange_data(
        self,
        download_file_id,
        extract_download_file_name,
        put_data_dir,
    ):
        download_data_file_name = extract_download_file_name + ".zip"
        download_data_path = os.path.join(put_data_dir,
                                          download_data_file_name)
        extract_data_path = os.path.join(put_data_dir,
                                         extract_download_file_name)
        if not self._exist_judgement(extract_data_path):
            logging.info(
                ("Downloading the data.....").format(download_data_path))
            download_file_from_google_drive(file_id=download_file_id,
                                            root=put_data_dir,
                                            filename=download_data_file_name)
            extract_archive(from_path=download_data_path,
                            to_path=put_data_dir,
                            remove_finished=True)

    def _exist_judgement(self, target_path):
        """ Judeg whether the input file/dir existed and whether it contains useful data """
        if not os.path.exists(target_path):
            logging.info(("The path {} does not exist").format(target_path))
            return False

        if os.path.isdir(target_path):
            if len(os.listdir(target_path)) == 0:
                logging.info(("The path {} does exist but contains no data"
                              ).format(target_path))
                return False
            else:
                return True

        logging.info(("The file {} does exist ").format(target_path))
        return True

    def num_modalities(self) -> int:
        return len(self.modality_names)
