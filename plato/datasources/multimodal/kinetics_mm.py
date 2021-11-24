"""
The backup interface for the kinetics dataset for a easier way to prepare the dataset

The data structure is:
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
"""

import re

import logging
import os
import shutil

from mmaction.tools.data.kinetics import download as kinetics_downloader

from plato.config import Config
from plato.datasources.multimodal import multimodal_base


class DataSource(multimodal_base.MultiModalDataSource):
    """The Gym dataset."""
    def __init__(self):
        super().__init__()

        self.data_name = Config().data.datasource
        base_data_name = re.findall(r'\D+', self.data_name)[0]

        # the rawframes contains the "flow", "rgb"
        # thus, the flow and rgb will be put in in same directory rawframes/
        self.modality_names = ["video", "audio", "rawframes", "audio_feature"]

        _path = Config().data.data_path
        self._data_path_process(data_path=_path, base_data_name=self.data_name)
        self._create_modalities_path(modality_names=self.modality_names)

        base_data_path = self.mm_data_info["base_data_dir_path"]
        # define all the dir here
        kinetics_anno_dir_name = "annotations"
        self.data_anno_dir_path = os.path.join(base_data_path,
                                               kinetics_anno_dir_name)

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
            new_file_name = base_data_name + "_" + file_name
            shutil.move(os.path.join(download_anno_path, file_name),
                        os.path.join(self.data_anno_dir_path, new_file_name))

        # download the trainset
        for split in ["train", "test", "validation"]:
            split_anno_path = os.path.join(
                self.data_anno_dir_path, base_data_name + "_" + split + ".csv")
            split_name = split if split != "validation" else "val"
            video_dir = os.path.join(base_data_path, "video_" + split_name)
            if not self._exist_judgement(video_dir):
                logging.info(
                    "Downloading the raw videos for the %s dataset. This may take a long time.",
                    self.data_name)
                kinetics_downloader.main(input_csv=split_anno_path,
                                         output_dir=video_dir,
                                         trim_format='%06d',
                                         num_jobs=2,
                                         tmp_dir='/tmp/kinetics')
        logging.info("Done.")

        logging.info("The %s dataset has been prepared", self.data_name)
