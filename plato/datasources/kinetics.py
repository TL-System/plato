"""

The Kinetics700 dataset.

Note that the setting for the data loader is obtained from the github
repo provided by the official workers:
https://github.com/pytorch/vision/references/video_classification/train.py

We consider three modalities: RGB, optical flow and audio.
    For RGB and flow, we use input clips of 16×224×224 as input.
        We follow [1] for visual pre-processing and augmentation.
    For audio, we use log-Mel with 100 temporal frames by 40 Mel filters.

    Audio and visual are temporally aligned.

[1]. Video classification with channel-separated convolutional networks.
    In ICCV, 2019. (CSN network)
    This is actually the csn network in the mmaction packet.

Also, the implementation of our code is based on the mmaction2 of the
  openmmlab https://openmmlab.com/.

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
import json
import logging
import os
import shutil

import torch

import pandas as pd

from mmaction.tools.data.kinetics import download as kinetics_downloader

from mmaction.datasets import build_dataset

from plato.config import Config
from plato.datasources import multimodal_base
from plato.datasources.datalib import frames_extraction_tools
from plato.datasources.datalib import audio_extraction_tools
from plato.datasources.datalib import modality_data_anntation_tools
from plato.datasources.datalib import data_utils


def creat_tiny_kinetics(anno_file_path, num_fist_videos):
    """ Create the tiny kinetics by using the first {num_fist_videos} 
         in the {anno_file}.
    """
    anno_file_base_dir = os.path.dirname(anno_file_path)
    anno_file_base_name, ext_type = os.path.basename(anno_file_path).split(".")

    anno_df = pd.read_csv(anno_file_path)
    tiny_annos = anno_df[:num_fist_videos]
    tiny_anno_file_path = os.path.join(
        anno_file_base_dir, anno_file_base_name + "_tiny." + ext_type)
    tiny_annos.to_csv(path_or_buf=tiny_anno_file_path, index=False)

    return tiny_anno_file_path


class KineticsDataset(multimodal_base.MultiModalDataset):
    """ Prepare the Flickr30K Entities dataset."""
    def __init__(self,
                 dataset_info,
                 phase,
                 phase_split,
                 data_types,
                 modality_sampler=None,
                 transform_image_dec_func=None,
                 transform_text_func=None):
        super().__init__()
        self.phase = phase
        self.phase_data_record = dataset_info
        self.phase_split = phase_split
        self.data_types = data_types
        self.transform_image_dec_func = transform_image_dec_func
        self.transform_text_func = transform_text_func

        self.phase_samples_name = list(self.phase_data_record.keys())

        self.supported_modalities = ["rgb", "flow", "audio"]

        # default utilizing the full modalities
        if modality_sampler is None:
            self.modality_sampler = self.supported_modalities
        else:
            self.modality_sampler = modality_sampler

    def __len__(self):
        return len(self.phase_data_record)

    def get_one_multimodal_sample(self, sample_idx):
        """ Obtain one sample from the Kinetics dataset. """
        pass


class DataSource(multimodal_base.MultiModalDataSource):
    """The Gym dataset."""
    def __init__(self):
        super().__init__()

        self.data_name = Config().data.datasource
        base_data_name = re.findall(r'\D+', self.data_name)[0]

        # The rawframes contains the "flow", "rgb"
        # thus, the flow and rgb will be put in the same directory rawframes/
        self.modality_names = [
            "video", "audio", "rgb", "flow", "audio_feature"
        ]
        # alternative: ["video", "audio", "rawframes", "audio_feature"]

        _path = Config().data.data_path
        # Generate the basic path for the dataset, it performs:
        #   1.- Assign path to self.mm_data_info
        #   2.- Assign splits path to self.splits_info
        #       where the root path for splits is the base_data_dir_path
        #       in self.mm_data_info
        self._data_path_process(data_path=_path, base_data_name=self.data_name)
        # Generate the modalities path for all splits, it performs:
        #   1.- Add modality path to each modality, the key style is:
        #       {modality}_path: ...
        #   Note: the rgb and flow modalities are merged into 'rawframes_path'
        #    as they belong to the same prototype "rawframes".
        self._create_modalities_path(modality_names=self.modality_names)

        # Set the annotation file path
        base_data_path = self.mm_data_info["base_data_dir_path"]

        # Define all the dir here
        kinetics_anno_dir_name = "annotations"
        self.data_anno_dir_path = os.path.join(base_data_path,
                                               kinetics_anno_dir_name)

        for split in ["train", "test", "validate"]:
            split_anno_path = os.path.join(
                self.data_anno_dir_path, base_data_name + "_" + split + ".csv")
            split_tiny_anno_path = os.path.join(
                self.data_anno_dir_path,
                base_data_name + "_" + split + "_tiny.csv")
            split_name = split if split != "validate" else "val"
            self.splits_info[split_name]["split_anno_file"] = split_anno_path
            self.splits_info[split_name][
                "split_tiny_anno_file"] = split_tiny_anno_path

        # Thus, after operating the above two functions,
        #  the self.splits_info can contain
        #   e.g. {'train':
        #       'path': xxx,
        #       'split_anno_file': xxx,
        #       'split_tiny_anno_file': xxx,
        #       'rawframes_path': xxx,
        #       'video_path': xxx}
        #  the self.mm_data_info can contain
        #   - source_data_path
        #   - base_data_dir_path

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

        # Whether to create the tiny dataset
        for split in ["train", "test", "validate"]:
            split_name = split if split != "validate" else "val"
            split_anno_path = self.splits_info[split_name]["split_anno_file"]
            if hasattr(Config().data, 'tiny_data') and Config().data.tiny_data:
                creat_tiny_kinetics(
                    anno_file_path=split_anno_path,
                    num_fist_videos=Config().data.tiny_data_number)

        # Download the raw datasets for splits
        for split in ["train", "test", "val"]:
            # Download the full dataset
            if hasattr(Config().data, 'tiny_data') and Config().data.tiny_data:
                split_anno_path = self.splits_info[split][
                    "split_tiny_anno_file"]
            else:  # Download the tiny dataset
                split_anno_path = self.splits_info[split]["split_anno_file"]
            video_path_format = self.set_modality_path_format(
                modality_name="video")
            video_dir = self.splits_info[split][video_path_format]
            if not self._exist_judgement(video_dir):
                num_workers = Config().data.downloader.num_workers
                # Set the tmp_dir to save the raw video
                # Then, the raw video will be clipped to save to
                #  the target video_dir

                tmp_dir = os.path.join(video_dir, "tmp")
                logging.info(
                    "Downloading the raw videos for the %s dataset. This may take a long time.",
                    self.data_name)
                kinetics_downloader.main(input_csv=split_anno_path,
                                         output_dir=video_dir,
                                         trim_format='%06d',
                                         num_jobs=num_workers,
                                         tmp_dir=tmp_dir)
        logging.info("Done.")

        logging.info("The %s dataset has been prepared", self.data_name)

        # Extract the splits information into the corresponding files
        # for split_name in ["train", "test", "validation"]:
        #     self.extract_split_list_files(mode=split_name)

    def extract_split_list_files(self, data_format, mode):
        """ Extract and generate the split information of current mode/phase """
        data_src_path_name = self.set_modality_path_format(data_format)
        gen_annots_op = modality_data_anntation_tools.GenerateMDataAnnotation(
            data_src_dir=self.splits_info[mode][data_src_path_name],
            data_annos_files_info=self.
            data_splits_file_info,  # a dict that contains the data splits' file path
            dataset_name=self.dataset_name,
            data_format=data_format,  # 'rawframes', 'videos'
            out_path=self.mm_data_info["base_data_dir_path"],
        )
        gen_annots_op.generate_data_splits_info_file(data_name=self.data_name)

        gen_annots_op = modality_data_anntation_tools.GenerateMDataAnnotation(
            data_src_dir=self.splits_info[mode]["rawframes_path"],
            data_annos_files_info=self.
            data_splits_file_info,  # a dict that contains the data splits' file path
            dataset_name=self.dataset_name,
            data_format="rawframes",  # 'rawframes', 'videos'
            out_path=self.mm_data_info["base_data_dir_path"],
        )
        gen_annots_op.generate_data_splits_info_file(data_name=self.data_name)

    def get_modality_name(self):
        """ Get all supports modalities """
        return ["rgb", "flow", "audio"]

    def extract_videos_rgb_flow_audio(self, mode="train"):
        """ Extract rgb, optical flow, and audio from videos """
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

    def get_phase_data_info(self, phase):
        """ Obtain the data information for the required phrase """
        path = self.splits_info[phase]["path"]
        save_path = os.path.join(path, phase + "_integrated_data.json")
        with open(save_path, 'r') as outfile:
            phase_data_info = json.load(outfile)
        return phase_data_info

    def get_phase_dataset(self, phase, modality_sampler):
        """ Obtain the dataset for the specific phase """
        phase_data_info = self.get_phase_data_info(phase)
        phase_split_info = self.splits_info[phase]
        dataset = KineticsDataset(dataset_info=phase_data_info,
                                  phase_split=phase_split_info,
                                  data_types=self.data_types,
                                  phase=phase,
                                  modality_sampler=modality_sampler)
        return dataset

    def get_train_set(self, modality_sampler):
        """ Get the train dataset """
        modality_dataset = []
        if "rgb" in modality_sampler:
            train_rgb_config = Config(
            ).data.multi_modal_pipeliner.rgb.config.train
            train_rgb_config = data_utils.dict_list2tuple(train_rgb_config)
            rgb_train_dataset = build_dataset(train_rgb_config)

            modality_dataset.append(rgb_train_dataset)
        if "flow" in modality_sampler:
            train_flow_config = Config(
            ).data.multi_modal_pipeliner.flow.config.train
            train_flow_config = data_utils.dict_list2tuple(train_flow_config)
            flow_train_dataset = build_dataset(train_flow_config)

            modality_dataset.append(flow_train_dataset)
        if "audio" in modality_sampler:
            train_audio_config = Config(
            ).data.multi_modal_pipeliner.audio.config.train
            train_audio_config = data_utils.dict_list2tuple(train_audio_config)
            audio_feature_train_dataset = build_dataset(train_audio_config)

            modality_dataset.append(audio_feature_train_dataset)

        mm_train_dataset = multimodal_base.MultiModalDataset(modality_dataset)
        return mm_train_dataset
