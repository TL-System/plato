#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import json
import logging
import os
import sys
import shutil

import torch
from torch.utils.data.dataloader import default_collate
from torchvision import datasets
from mmaction.tools.data.kinetics import download as kinetics_downloader
from mmaction.tools.data.gym import download as gym_downloader

from plato.config import Config
from plato.datasources.multimodal import multimodal_base
'''
├── data
│   ├── ${DATASET}
│   │   ├── ${DATASET}_train_list_videos.txt
│   │   ├── ${DATASET}_val_list_videos.txt
│   │   ├── annotations
│   │   ├── videos_train
│   │   ├── videos_val
│   │   │   ├── abseiling
│   │   │   │   ├── 0wR5jVB-WPk_000417_000427.mp4
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── wrapping_present
│   │   │   ├── ...
│   │   │   ├── zumba
│   │   ├── rawframes_train
│   │   ├── rawframes_val
'''


class DataSource(multimodal_base.MultiModalDataSource):
    """The Gym dataset."""
    def __init__(self):
        super().__init__()

        self.data_name = Config().data.datasource

        # the rawframes contains the "flow", "rgb"
        # thus, the flow and rgb will be put in in same directory rawframes/
        self.modality_names = ["video", "audio", "rawframes", "audio_feature"]

        _path = Config().data.data_path
        self._data_path_process(data_path=_path, base_data_name=self.data_name)
        self._create_modalities_path(modality_names=self.modality_names)

        base_data_path = self.mm_data_info["base_data_dir_path"]
        # define all the dir here
        Kinetics_annotation_dir_name = "annotations"
        self.data_anno_dir_path = os.path.join(base_data_path,
                                               Kinetics_annotation_dir_name)
        self.data_anno_file_path = os.path.join(self.data_anno_dir_path,
                                                "annotation.json")
        self.raw_videos_path = os.path.join(base_data_path, "videos")
        self.event_dir_path = os.path.join(base_data_path, "event")
        self.event_subsection_dir_path = os.path.join(base_data_path,
                                                      "subactions")
        self.data_event_anno_file_path = os.path.join(self.data_anno_dir_path,
                                                      "event_annotation.json")
        self.event_subsection_frames_dir_path = os.path.join(
            base_data_path, "subaction_frames")
        self.event_subsection_audios_dir_path = os.path.join(
            base_data_path, "subaction_audios")

        self.event_subsection_audios_fea_dir_path = os.path.join(
            base_data_path, "subaction_audios_features")

        anno_download_url = (
            "https://storage.googleapis.com/deepmind-media/Datasets/{}.tar.gz"
        ).format(self.data_name)

        extracted_anno_file_name = self._download_arrange_data(
            download_url_address=anno_download_url,
            put_data_dir=self.data_anno_dir_path,
            obtained_file_name=None)
        download_anno_path = os.path.join(self.data_anno_dir_path,
                                          extracted_anno_file_name)

        downloaded_files = os.listdir(download_anno_path)
        for file_name in downloaded_files:

            shutil.move(os.path.join(download_anno_path, file),
                        os.path.join(download_anno_path, file))
        if not self._exist_judgement(self.raw_videos_path):

            logging.info(
                "Downloading the raw videos for the Gym dataset. This may take a long time."
            )

            logging.info("Done.")

        logging.info(
            ("The {} dataset has been prepared").format(self.data_name))
