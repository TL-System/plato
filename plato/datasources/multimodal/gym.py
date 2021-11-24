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

import torch

from mmaction.tools.data.gym import download as gym_downloader

from plato.config import Config
from plato.datasources.datalib.gym_utils import gym_trim
from plato.datasources.multimodal import multimodal_base
from plato.datasources.datalib import frames_extraction_tools
from plato.datasources.datalib import audio_extraction_tools


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
        kinetics_anno_dir_name = "annotations"
        self.data_anno_dir_path = os.path.join(base_data_path,
                                               kinetics_anno_dir_name)
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

        anno_url = "https://sdolivia.github.io/FineGym/resources/ \
                    dataset/finegym_annotation_info_v1.0.json"

        train_url = "https://sdolivia.github.io/FineGym/resources/ \
                    dataset/gym99_train_element_v1.0.txt"

        eval_url = "https://sdolivia.github.io/FineGym/resources/dataset/gym99_val_element.txt"

        _ = self._download_arrange_data(download_url_address=anno_url,
                                        put_data_dir=self.data_anno_dir_path,
                                        obtained_file_name="annotation.json")

        _ = self._download_arrange_data(
            download_url_address=train_url,
            put_data_dir=self.data_anno_dir_path,
            obtained_file_name="gym99_train_org.txt")

        _ = self._download_arrange_data(download_url_address=eval_url,
                                        put_data_dir=self.data_anno_dir_path,
                                        obtained_file_name="gym99_val_org.txt")

        if not self._exist_judgement(self.raw_videos_path):

            logging.info(
                "Downloading the raw videos for the Gym dataset. This may take a long time."
            )

            logging.info("Done.")

            gym_downloader.main(input=self.data_anno_file_path,
                                output_dir=self.raw_videos_path,
                                num_jobs=Config().data.num_workers)

        # Trim Videos into Events
        if not self._exist_judgement(self.event_dir_path):
            gym_trim.trim_event(video_root=self.raw_videos_path,
                                anno_file=self.data_anno_file_path,
                                event_anno_file=self.data_event_anno_file_path,
                                event_root=self.event_dir_path)
        if not self._exist_judgement(self.event_subsection_dir_path):
            gym_trim.trim_subsection(
                event_anno_file=self.data_event_anno_file_path,
                event_root=self.event_dir_path,
                subaction_root=self.event_subsection_dir_path)

        logging.info("The Gym dataset has been prepared")

    def extract_videos_rgb_flow_audio(self):
        """ Extract the rgb optical flow audios from the video """
        src_videos_dir = self.event_subsection_dir_path
        frames_out_dir_path = self.event_subsection_frames_dir_path
        rgb_out_dir_path = self.event_subsection_frames_dir_path
        flow_our_dir_path = self.event_subsection_frames_dir_path
        audio_out_dir_path = self.event_subsection_audios_dir_path
        # audio_feature_dir_path = self.event_subsection_audios_fea_dir_path

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
            if not self._exist_judgement(
                    rgb_out_dir_path) and not self._exist_judgement(
                        flow_our_dir_path):
                vdf_extractor.build_full_frames_gpu(
                    to_dir_path=frames_out_dir_path,
                    new_short=1,
                    new_width=0,
                    new_height=0)
        else:
            if not self._exist_judgement(frames_out_dir_path):
                vdf_extractor.build_frames_cpu(to_dir=frames_out_dir_path)

        if not self._exist_judgement(audio_out_dir_path):
            vda_extractor.build_audios(to_dir=audio_out_dir_path)

        # split the data based on the frames information
        gym_trim.generate_splits_list(data_root=self.event_subsection_dir_path,
                                      annotation_root=self.data_anno_dir_path,
                                      frame_data_root=frames_out_dir_path)
