"""
The Gym dataset.

Note that the setting for the data loader is obtained from the github repo provided
by the official workers:
    Finegym: A hierarchical video dataset for fine-grained action understanding

The data structure should be:

├── data
│   ├── gym99
|   |   ├── annotations
|   |   |   ├── gym99_train_org.txt
|   |   |   ├── gym99_val_org.txt
|   |   |   ├── gym99_train.txt
|   |   |   ├── gym99_val.txt
|   |   |   ├── annotation.json
|   |   |   └── event_annotation.json
│   │   ├── videos
|   |   |   ├── 0LtLS9wROrk.mp4
|   |   |   ├── ...
|   |   |   └── zfqS-wCJSsw.mp4
│   │   ├── events
|   |   |   ├── 0LtLS9wROrk_E_002407_002435.mp4
|   |   |   ├── ...
|   |   |   └── zfqS-wCJSsw_E_006732_006824.mp4
│   │   ├── subactions
|   |   |   ├── 0LtLS9wROrk_E_002407_002435_A_0003_0005.mp4
|   |   |   ├── ...
|   |   |   └── zfqS-wCJSsw_E_006244_006252_A_0000_0007.mp4
|   |   └── subaction_frames
|   |   |── subaction_audios

"""

import logging
import os
import shutil

import torch

from mmaction.tools.data.gym import download as gym_downloader
from mmaction.datasets import build_dataset

from plato.config import Config
from plato.datasources.datalib.gym_utils import gym_trim
from plato.datasources import multimodal_base
from plato.datasources.datalib import frames_extraction_tools
from plato.datasources.datalib import audio_extraction_tools
from plato.datasources.datalib import data_utils


class GymDataset(multimodal_base.MultiModalDataset):
    """ Prepare the Gym dataset."""

    def __init__(self,
                 multimodal_data_holder,
                 phase,
                 phase_info,
                 modality_sampler=None):
        super().__init__()
        self.phase = phase
        #  multimodal_data_holder is a dict:
        #    {"rgb": rgb_dataset, "flow": flow_dataset, "audio": audio_dataset}
        self.phase_multimodal_data_record = multimodal_data_holder

        # a dict presented as:
        #   "rgb": <rgb_annotation_file_path>
        self.phase_info = phase_info

        self.modalities_name = list(multimodal_data_holder.keys())

        self.supported_modalities = ["rgb", "flow", "audio_feature"]

        # default utilizing the full modalities
        if modality_sampler is None:
            self.modality_sampler = self.supported_modalities
        else:
            self.modality_sampler = modality_sampler

        self.targets = self.get_targets()

    def __len__(self):
        return len(self.phase_multimodal_data_record)

    def get_targets(self):
        """ Obtain the labels of samples in current phase dataset.  """
        # There is no label provided in the fine gym dataset currently
        #  This part will be added afterward
        return [0]

    def get_one_multimodal_sample(self, sample_idx):
        """ Obtain one sample from the Kinetics dataset. """
        obtained_mm_sample = dict()

        for modality_name in self.modalities_name:

            modality_dataset = self.phase_multimodal_data_record[modality_name]
            obtained_mm_sample[modality_name] = modality_dataset[sample_idx]

        return obtained_mm_sample


class DataSource(multimodal_base.MultiModalDataSource):
    """The Gym dataset."""

    def __init__(self, **kwargs):
        super().__init__()

        self.data_name = Config().data.datasource

        # the rawframes contains the "flow", "rgb"
        # thus, the flow and rgb will be put in in same directory rawframes/
        # self.modality_names = ["video", "audio", "rawframes", "audio_feature"]
        self.modality_names = [
            "video", "audio", "rgb", "flow", "audio_feature"
        ]

        _path = Config().params['data_path']
        self._data_path_process(data_path=_path, base_data_name=self.data_name)
        self._create_modalities_path(modality_names=self.modality_names)

        base_data_path = self.mm_data_info["data_path"]
        # define all the dir here
        gym_anno_dir_name = "annotations"
        self.data_annotation_path = os.path.join(base_data_path,
                                                 gym_anno_dir_name)

        self.data_anno_file_path = os.path.join(self.data_annotation_path,
                                                "annotation.json")
        self.categoty_anno_file_path = os.path.join(self.data_annotation_path,
                                                    "gym99_categories.txt")

        self.raw_videos_path = os.path.join(base_data_path, "videos")
        self.event__path = os.path.join(base_data_path, "event")
        self.event_subsection__path = os.path.join(base_data_path,
                                                   "subactions")
        self.data_event_anno_file_path = os.path.join(
            self.data_annotation_path, "event_annotation.json")
        self.event_subsection_frames__path = os.path.join(
            base_data_path, "subaction_rawframes")
        self.event_subsection_audios__path = os.path.join(
            base_data_path, "subaction_audios")

        self.event_subsection_audios_fea__path = os.path.join(
            base_data_path, "subaction_audios_features")

        self.rawframes_splits_list_files_into = {
            "train":
            os.path.join(self.data_annotation_path,
                         "gym99_train_rawframes.txt"),
            "val":
            os.path.join(self.data_annotation_path, "gym99_val_rawframes.txt")
        }

        self.audios_splits_list_files_into = {
            "train":
            os.path.join(self.data_annotation_path, "gym99_train_audios.txt"),
            "val":
            os.path.join(self.data_annotation_path, "gym99_val_audios.txt")
        }
        self.audio_features_splits_list_files_into = {
            "train":
            os.path.join(self.data_annotation_path,
                         "gym99_train_audio_features.txt"),
            "val":
            os.path.join(self.data_annotation_path,
                         "gym99_val_audio_features.txt")
        }

        set_level_category_url = "https://sdolivia.github.io/FineGym/resources/dataset/set_categories.txt"
        g99_categoty_url = "https://sdolivia.github.io/FineGym/resources/dataset/gym99_categories.txt"

        anno_url = "https://sdolivia.github.io/FineGym/resources/dataset/finegym_annotation_info_v1.0.json"

        train_url = "https://sdolivia.github.io/FineGym/resources/dataset/gym99_train_element_v1.0.txt"

        eval_url = "https://sdolivia.github.io/FineGym/resources/dataset/gym99_val_element.txt"

        _ = self._download_arrange_data(
            download_url_address=set_level_category_url,
            data_path=self.data_annotation_path,
            obtained_file_name="set_categories.txt")

        _ = self._download_arrange_data(
            download_url_address=g99_categoty_url,
            data_path=self.data_annotation_path,
            obtained_file_name="gym99_categories.txt")

        _ = self._download_arrange_data(download_url_address=anno_url,
                                        data_path=self.data_annotation_path,
                                        obtained_file_name="annotation.json")

        _ = self._download_arrange_data(
            download_url_address=train_url,
            data_path=self.data_annotation_path,
            obtained_file_name="gym99_train_org.txt")

        _ = self._download_arrange_data(download_url_address=eval_url,
                                        data_path=self.data_annotation_path,
                                        obtained_file_name="gym99_val_org.txt")

        if not self._exists(self.raw_videos_path):

            logging.info(
                "Downloading the raw videos for the Gym dataset. This may take a long time."
            )

            gym_downloader.main(input=self.data_anno_file_path,
                                output_dir=self.raw_videos_path,
                                num_jobs=Config().data.downloader.num_workers)
            logging.info("Done.")

        # Trim Videos into Events
        if not self._exists(self.event__path):
            gym_trim.trim_event(video_root=self.raw_videos_path,
                                anno_file=self.data_anno_file_path,
                                event_anno_file=self.data_event_anno_file_path,
                                event_root=self.event__path)
        if not self._exists(self.event_subsection__path):
            gym_trim.trim_subsection(
                event_anno_file=self.data_event_anno_file_path,
                event_root=self.event__path,
                subaction_root=self.event_subsection__path)

        logging.info("The Gym dataset has been prepared")
        self.extract_videos_rgb_flow_audio()

    def extract_videos_rgb_flow_audio(self):
        """ Extract the rgb optical flow audios from the video """
        src_videos_dir = self.event_subsection__path
        frames_out__path = self.event_subsection_frames__path
        rgb_out__path = self.event_subsection_frames__path
        flow_our__path = self.event_subsection_frames__path
        audio_out__path = self.event_subsection_audios__path
        audio_feature__path = self.event_subsection_audios_fea__path

        # define the modalities extractor
        vdf_extractor = frames_extraction_tools.VideoFramesExtractor(
            video_src_dir=src_videos_dir,
            dir_level=1,
            num_worker=8,
            video_ext="mp4",
            mixed_ext=False)
        vda_extractor = audio_extraction_tools.VideoAudioExtractor(
            video_src_dir=src_videos_dir,
            dir_level=1,
            num_worker=8,
            video_ext="mp4",
            mixed_ext=False)

        if torch.cuda.is_available():
            if not self._exists(rgb_out__path) and not self._exists(
                    flow_our__path):
                logging.info(
                    "Extracting frames by GPU from videos in %s to %s.",
                    src_videos_dir, rgb_out__path)
                vdf_extractor.build_full_frames_gpu(to__path=frames_out__path,
                                                    new_short=256,
                                                    new_width=0,
                                                    new_height=0)
        else:
            if not self._exists(rgb_out__path):
                logging.info(
                    "Extracting frames by CPU from videos in %s to %s.",
                    src_videos_dir, rgb_out__path)
                vdf_extractor.build_frames_cpu(to_dir=frames_out__path)

        if not self._exists(audio_out__path):
            logging.info("Extracting audios by CPU from videos in %s to %s.",
                         src_videos_dir, audio_out__path)
            vda_extractor.build_audios(to_dir=audio_out__path)

        if not self._exists(audio_feature__path):
            logging.info(
                "Extracting audios feature by CPU from audios in %s to %s.",
                audio_out__path, audio_feature__path)
            # # window_size:32ms hop_size:16ms

            vda_extractor.build_audios_features(
                audio_src_path=audio_out__path,
                to_dir=audio_feature__path,
                fft_size=512,  # fft_size / sample_rate is window size
                hop_size=256)
        # extract the splits data into list files based on the frames information
        gym_trim.generate_splits_list(
            data_root=self.event_subsection__path,
            annotation_root=self.data_annotation_path,
            frame_data_root=frames_out__path)

        # generate the audio and audio features splits file
        # just copy the frame files to the audio ones
        for split in list(self.rawframes_splits_list_files_into.keys()):
            rawframes_split_file_path = self.rawframes_splits_list_files_into[
                split]
            audios_split_file_path = self.audios_splits_list_files_into[split]
            audio_features_split_file_path = self.audios_splits_list_files_into[
                split]
            shutil.copy(src=rawframes_split_file_path,
                        dst=audios_split_file_path)
            shutil.copy(src=rawframes_split_file_path,
                        dst=audio_features_split_file_path)

    def correct_current_config(self, loaded_plato_config, mode, modality_name):
        """ Correct the loaded configuration settings based on
            on-hand data information """

        # 1.1. convert plato config to dict type
        loaded_config = data_utils.config_to_dict(loaded_plato_config)
        # 1.2. convert the list to tuple
        loaded_config = data_utils.dict_list2tuple(loaded_config)

        # 2. using the obtained annotation file replace the user set ones
        #   in the configuration file
        #   The main reason is that the obtained path here is the full path
        cur_rawframes_anno_file_path = self.rawframes_splits_list_files_into[
            mode]
        cur_rawframes_data_path = self.event_subsection_frames__path
        cur_videos_anno_file_path = None
        cur_video_data_path = self.event_subsection__path
        cur_audio_feas_anno_file_path = self.audios_splits_list_files_into[
            mode]
        cur_audio_feas_data_path = self.event_subsection_audios__path

        if modality_name == "rgb" or modality_name == "flow":
            loaded_config["ann_file"] = cur_rawframes_anno_file_path
        elif modality_name == "audio_feature":
            loaded_config["ann_file"] = cur_audio_feas_anno_file_path
        else:
            loaded_config["ann_file"] = cur_videos_anno_file_path

        # 3. reset the data_prefix by using the modality path
        if modality_name == "rgb" or modality_name == "flow":
            loaded_config["data_prefix"] = cur_rawframes_data_path
        elif modality_name == "audio_feature":
            loaded_config["data_prefix"] = cur_audio_feas_data_path
        else:
            loaded_config["data_prefix"] = cur_video_data_path

        return loaded_config

    def get_phase_dataset(self, phase, modality_sampler):
        """ Get the dataset for the specific phase. """
        rgb_mode_config = getattr(Config().data.multi_modal_configs.rgb, phase)
        flow_mode_config = getattr(Config().data.multi_modal_configs.flow,
                                   phase)
        audio_feature_mode_config = getattr(
            Config().data.multi_modal_configs.audio_feature, phase)

        rgb_mode_config = self.correct_current_config(
            loaded_plato_config=rgb_mode_config,
            mode=phase,
            modality_name="rgb")
        flow_mode_config = self.correct_current_config(
            loaded_plato_config=flow_mode_config,
            mode=phase,
            modality_name="flow")
        audio_feature_mode_config = self.correct_current_config(
            loaded_plato_config=audio_feature_mode_config,
            mode=phase,
            modality_name="audio_feature")
        # build a RawframeDataset
        rgb_mode_dataset = build_dataset(rgb_mode_config)
        flow_mode_dataset = build_dataset(flow_mode_config)
        audio_feature_mode_dataset = build_dataset(audio_feature_mode_config)

        multi_modal_mode_data = {
            "rgb": rgb_mode_dataset,
            "flow": flow_mode_dataset,
            "audio_feature": audio_feature_mode_dataset
        }

        multi_modal_mode_info = {
            "rgb": rgb_mode_config["ann_file"],
            "flow": flow_mode_config["ann_file"],
            "audio_feature": audio_feature_mode_config["ann_file"],
            "categories": self.categoty_anno_file_path
        }

        gym_mode_dataset = GymDataset(
            multimodal_data_holder=multi_modal_mode_data,
            phase="train",
            phase_info=multi_modal_mode_info,
            modality_sampler=modality_sampler)

        return gym_mode_dataset

    def get_train_set(self, modality_sampler=None):
        """ Obtain the trainset for multimodal data. """
        gym_train_dataset = self.get_phase_dataset(
            phase="train", modality_sampler=modality_sampler)

        return gym_train_dataset

    def get_test_set(self, modality_sampler=None):
        """ Obtain the testset for multimodal data.

            Note, in the kinetics dataset, there is no testset in which
             samples contain the groundtruth label.
             Thus, we utilize the validation set directly.
        """
        gym_val_dataset = self.get_phase_dataset(
            phase="val", modality_sampler=modality_sampler)

        return gym_val_dataset

    def get_modality_name(self):
        """ Get all supports modalities """
        return ["rgb", "flow", "audio"]
