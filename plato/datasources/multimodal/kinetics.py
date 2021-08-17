"""
The Kinetics700 dataset.

Note that the setting for the data loader is obtained from the github repo provided by the official workers:
https://github.com/pytorch/vision/references/video_classification/train.py
"""

import logging
import os
import sys
import shutil

import torch
from torch.utils.data.dataloader import default_collate
from torchvision import datasets
from mmaction.datasets import build_dataset
from mmaction.datasets import rawframe_dataset
from mmaction.datasets import audio_feature_dataset

from plato.config import Config
from plato.datasources.multimodal import multimodal_base

from plato.datasources.datalib.kinetics_utils import download_tools
from plato.datasources.datalib.kinetics_utils import utils as kine_utils
from plato.datasources.datalib import video_transform
from plato.datasources.datalib import frames_extraction_tools
from plato.datasources.datalib import audio_extraction_tools
from plato.datasources.datalib import modality_data_anntation_tools
""" 
We consider three modalities: RGB, optical flow and audio. 
    - For RGB and flow, we use input clips of 16×224×224 as input. We fol- low [1] for visual pre-processing and augmentation. 
    - For audio, we use log-Mel with 100 temporal frames by 40 Mel filters. Audio and visual are temporally aligned.

[1]. Video classification with channel-separated convolutional networks. In ICCV, 2019. (CSN network)
    This is actually the csn network in the mmaction packet

"""


class DataSource(multimodal_base.MultiModalDataSource):
    """The Kinetics700 dataset."""
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
        download_url = Config().data.download_url
        extracted_dir_name = self._download_arrange_data(
            download_url_address=download_url, put_data_dir=base_data_path)

        download_info_dir_path = os.path.join(base_data_path,
                                              extracted_dir_name)
        # convert the Kinetics data dir structure to the typical one shown in
        #   https://github.com/open-mmlab/mmaction2/blob/master/tools/data/kinetics/README.md
        Kinetics_annotation_dir_name = "annotations"
        ann_dst_path = os.path.join(base_data_path,
                                    Kinetics_annotation_dir_name)
        if not self._exist_judgement(ann_dst_path):
            shutil.copytree(download_info_dir_path, ann_dst_path)

        # obtain the path of the data information
        self.data_categories_file = os.path.join(base_data_path,
                                                 "categories.json")
        self.data_classes_file = os.path.join(base_data_path, "classes.json")
        self.train_info_data_path = os.path.join(download_info_dir_path,
                                                 "train.json")
        self.test_info_data_path = os.path.join(download_info_dir_path,
                                                "test.json")
        self.val_info_data_path = os.path.join(download_info_dir_path,
                                               "validate.json")

        self.data_splits_file_info = {
            "train": os.path.join(download_info_dir_path, "train.csv"),
            "test": os.path.join(download_info_dir_path, "test.csv"),
            "val": os.path.join(download_info_dir_path, "validate.csv")
        }

        self.data_classes = kine_utils.extract_data_classes(
            data_classes_file=self.data_classes_file,
            train_info_data_path=self.train_info_data_path,
            val_info_data_path=self.test_info_data_path)

        # get the download hyper-parameters
        num_workers = Config().data.num_workers
        failed_save_file = Config().data.failed_save_file
        compress = Config().data.compress
        verbose = Config().data.verbose
        skip = Config().data.skip
        log_file = Config().data.log_file

        failed_save_file = os.path.join(base_data_path, failed_save_file)

        # download the raw video dataset if necessary
        if not self._exist_judgement(self.splits_info["train"]["video_path"]):

            logging.info(
                "Downloading the raw videos for the Kinetics700 dataset. This may take a long time."
            )
            download_tools.download_train_val_sets(
                splits_info=zip(
                    [self.train_info_data_path, self.val_info_data_path], [
                        self.splits_info["train"]["video_path"],
                        self.splits_info["val"]["video_path"]
                    ]),
                data_classes=self.data_classes,
                data_categories_file=self.data_categories_file,
                num_workers=num_workers,
                failed_log=failed_save_file,
                compress=compress,
                verbose=verbose,
                skip=skip,
                log_file=log_file)

            download_tools.download_test_set(
                test_info_data_path=self.test_info_data_path,
                test_video_des_path=self.splits_info["test"]["video_path"],
                num_workers=num_workers,
                failed_log=failed_save_file,
                compress=compress,
                verbose=verbose,
                skip=skip,
                log_file=log_file)
            logging.info("Done.")

        logging.info("The Kinetics700 dataset has been prepared")

    def get_modality_name():
        return ["RGB", "Flow", "Audio"]

    def extract_videos_rgb_flow_audio(self, mode="train", device="CPU"):
        src_mode_videos_dir = os.path.join(
            self.splits_info[mode]["video_path"])
        rgb_out_dir_path = self.splits_info[mode]["rawframes_path"]
        flow_our_dir_path = self.splits_info[mode]["rawframes_path"]
        audio_out_dir_path = self.splits_info[mode]["audio_path"]
        audio_feature_dir_path = self.splits_info[mode]["audio_feature_path"]

        # define the modalities extractor
        vdf_extractor = frames_extraction_tools.VideoFramesExtractor(
            video_src_dir=src_mode_videos_dir,
            dir_level=2,
            num_worker=8,
            video_ext="mp4",
            mixed_ext=False)
        vda_extractor = audio_extraction_tools.VideoAudioExtractor(
            video_src_dir=src_mode_videos_dir,
            dir_level=2,
            num_worker=8,
            video_ext="mp4",
            mixed_ext=False)

        if torch.cuda.is_available():
            if not self._exist_judgement(
                    rgb_out_dir_path) and not self._exist_judgement(
                        flow_our_dir_path):
                vdf_extractor.build_frames_gpu(rgb_out_dir_path,
                                               flow_our_dir_path,
                                               new_short=1,
                                               new_width=0,
                                               new_height=0)
        else:
            if not self._exist_judgement(
                    self.splits_info[mode]["rawframes_path"]):
                vdf_extractor.build_frames_cpu(
                    to_dir=self.splits_info[mode]["rawframes_path"])

        if not self._exist_judgement(audio_out_dir_path):
            vda_extractor.build_audios(to_dir=audio_out_dir_path)

        if not self._exist_judgement(audio_feature_dir_path):
            vda_extractor.build_audios_features(
                audio_src_path=audio_out_dir_path,
                to_dir=audio_feature_dir_path)

    def extract_split_list_files(self):
        gen_annots_op = modality_data_anntation_tools.GenerateMDataAnnotation(
            data_src_dir=self.splits_info[mode]["rawframes_path"],
            data_annos_files_info=self.
            data_splits_file_info,  # a dict that contains the data splits' file path
            dataset_name=self.dataset_name,
            data_format="rawframes",  # 'rawframes', 'videos'
            out_path=self.
            mm_data_info["base_data_dir_path"],  # put to the base dir
        )
        gen_annots_op.generate_data_splits_info_file(data_name=self.data_name)

    def get_train_set(self, modality_sampler):
        modality_dataset = []
        if "RGB" in modality_sampler:
            rgb_train_dataset = build_dataset(
                Config().multimodal_data.rgb_data.train)
            modality_dataset.append(rgb_train_dataset)
        if "Flow" in modality_sampler:
            flow_train_dataset = build_dataset(
                Config().multimodal_data.flow_data.train)
            modality_dataset.append(flow_train_dataset)
        if "Audio" in modality_sampler:
            audio_feature_train_dataset = build_dataset(
                Config().multimodal_data.audio_data.train)
            modality_dataset.append(audio_feature_train_dataset)

        mm_train_dataset = multimodal_base.MultiModalDataset(modality_dataset)
        return mm_train_dataset

    def get_test_set(self):
        rgb_test_dataset = build_dataset(
            Config().multimodal_data.rgb_data.test)
        flow_test_dataset = build_dataset(
            Config().multimodal_data.flow_data.test)
        audio_feature_test_dataset = build_dataset(
            Config().multimodal_data.audio_data.test)

        mm_test_dataset = multimodal_base.MultiModalDataset(
            [rgb_test_dataset, flow_test_dataset, audio_feature_test_dataset])
        return mm_test_dataset

    def get_val_set(self):
        rgb_val_dataset = build_dataset(Config().multimodal_data.rgb_data.val)
        flow_val_dataset = build_dataset(
            Config().multimodal_data.flow_data.val)
        audio_feature_val_dataset = build_dataset(
            Config().multimodal_data.audio_data.val)

        # one sample of this dataset contains three part of data
        mm_val_dataset = multimodal_base.MultiModalDataset(
            [rgb_val_dataset, flow_val_dataset, audio_feature_val_dataset])
        return mm_val_dataset

    @staticmethod
    def get_data_loader(self, batch_size, dataset, sampler):
        def collate_fn(batch):
            return default_collate

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           pin_memory=True,
                                           collate_fn=collate_fn)
