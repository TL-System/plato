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


For data_formates, we support "videos", "rawframes", "audios", "audio_features"
for modality, we support "video", "audio", "audio_feature", "rgb", "flow"

"""

import re
import logging
import os
import shutil
from collections import defaultdict

import torch

from mmaction.tools.data.kinetics import download as kinetics_downloader
from mmaction.datasets import build_dataset

from plato.config import Config
from plato.datasources import multimodal_base
from plato.datasources.datalib import frames_extraction_tools
from plato.datasources.datalib import audio_extraction_tools
from plato.datasources.datalib import modality_data_anntation_tools
from plato.datasources.datalib import data_utils
from plato.datasources.datalib import tiny_data_tools


def obtain_required_anno_files(splits_info):
    """ Obtain the general full/tiny annotation files for splits """
    required_anno_files = {"train": '', "test": '', "val": ''}
    for split in ['train', 'test', 'val']:
        split_info = splits_info[split]
        # Obtain the annotation files for the whole dataset
        if hasattr(Config().data, 'tiny_data') and Config().data.tiny_data:
            split_anno_path = split_info["split_tiny_anno_file"]
        else:  # Obtain the annotation files for the tiny dataset
            split_anno_path = split_info["split_anno_file"]

        required_anno_files[split] = split_anno_path
    return required_anno_files


class KineticsDataset(multimodal_base.MultiModalDataset):
    """ Prepare the Kinetics dataset."""

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
        # the order of samples in rgb, flow, or audio annotation files
        #  is maintained the same, thus obtain either one is great.
        # Normally, rgb and flow belong to the rawframes
        rawframes_anno_list_file_path = self.phase_info["rgb"]
        annos_list = data_utils.read_anno_file(rawframes_anno_list_file_path)

        obtained_targets = [anno_item["label"][0] for anno_item in annos_list]

        return obtained_targets

    def get_one_multimodal_sample(self, sample_idx):
        """ Obtain one sample from the Kinetics dataset. """
        obtained_mm_sample = dict()

        for modality_name in self.modalities_name:

            modality_dataset = self.phase_multimodal_data_record[modality_name]
            obtained_mm_sample[modality_name] = modality_dataset[sample_idx]

        return obtained_mm_sample


class DataSource(multimodal_base.MultiModalDataSource):
    """The Kinetics datasource."""

    def __init__(self, **kwargs):
        super().__init__()

        self.data_name = Config().data.datasource
        base_data_name = re.findall(r'\D+', self.data_name)[0]

        # The rawframes contains the "flow", "rgb"
        # thus, the flow and rgb will be put in the same directory rawframes/
        self.modality_names = [
            "video", "audio", "rgb", "flow", "audio_feature"
        ]

        _path = Config().params['data_path']
        # Generate the basic path for the dataset, it performs:
        #   1.- Assign path to self.mm_data_info
        #   2.- Assign splits path to self.splits_info
        #       where the root path for splits is the data_path
        #       in self.mm_data_info
        self._data_path_process(data_path=_path, base_data_name=self.data_name)
        # Generate the modalities path for all splits, it performs:
        #   1.- Add modality path to each modality, the key style is:
        #       {modality}_path: ...
        #   Note: the rgb and flow modalities are merged into 'rawframes_path'
        #    as they belong to the same prototype "rawframes".
        self._create_modalities_path(modality_names=self.modality_names)

        # Set the annotation file path
        base_data_path = self.mm_data_info["data_path"]

        # Define all the dir here
        kinetics_anno_dir_name = "annotations"
        self.data_annotation_path = os.path.join(base_data_path,
                                                 kinetics_anno_dir_name)

        for split in ["train", "test", "validate"]:
            split_anno_path = os.path.join(
                self.data_annotation_path,
                base_data_name + "_" + split + ".csv")
            split_tiny_anno_path = os.path.join(
                self.data_annotation_path,
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
        #   - data_path

        anno_download_url = (
            "https://storage.googleapis.com/deepmind-media/Datasets/{}.tar.gz"
        ).format(self.data_name)

        extracted_anno_file_name = self._download_arrange_data(
            download_url_address=anno_download_url,
            data_path=self.data_annotation_path,
            obtained_file_name=None)
        download_anno_path = os.path.join(self.data_annotation_path,
                                          extracted_anno_file_name)

        downloaded_files = os.listdir(download_anno_path)
        for file_name in downloaded_files:
            new_file_name = base_data_name + "_" + file_name
            shutil.move(os.path.join(download_anno_path, file_name),
                        os.path.join(self.data_annotation_path, new_file_name))

        # Whether to create the tiny dataset
        for split in ["train", "test", "validate"]:
            split_name = split if split != "validate" else "val"
            split_anno_path = self.splits_info[split_name]["split_anno_file"]

        if hasattr(Config().data, 'tiny_data') and Config().data.tiny_data:
            anno_files_info = {
                "train": self.splits_info["train"]["split_anno_file"],
                "test": self.splits_info["test"]["split_anno_file"],
                "val": self.splits_info["val"]["split_anno_file"]
            }
            tiny_data_tools.create_tiny_kinetics_anno(
                kinetics_annotation_files_info=anno_files_info,
                num_samples=Config().data.tiny_data_number,
                random_seed=Config().data.random_seed)

        # Download the raw datasets for splits
        # There is no need to download data for test as the test dataset of kinetics does not
        #   contain labels.
        required_anno_files = obtain_required_anno_files(self.splits_info)
        for split in ["train", "val"]:
            split_anno_path = required_anno_files[split]
            video_path_format = self.set_modality_path_key_format(
                modality_name="video")
            video_dir = self.splits_info[split][video_path_format]
            if not self._exists(video_dir):
                num_workers = Config().data.downloader.num_workers
                # Set the tmp_dir to save the raw video
                # Then, the raw video will be clipped to save to
                #  the target video_dir

                tmp_dir = os.path.join(video_dir, "tmp")
                logging.info(
                    "Downloading the raw videos for the %s %s dataset. This may take a long time.",
                    self.data_name, split)
                kinetics_downloader.main(input_csv=split_anno_path,
                                         output_dir=video_dir,
                                         trim_format='%06d',
                                         num_jobs=num_workers,
                                         tmp_dir=tmp_dir)
        # Rename of class name
        for split in ["train", "val"]:
            self.rename_classes(mode=split)

        logging.info("Done.")

        logging.info("The %s dataset has been prepared", self.data_name)

        # Extract rgb, flow, audio, audio_feature from the video
        for split in ["train", "val"]:
            self.extract_videos_rgb_flow_audio(mode=split)

        # Extract the splits information into the
        #   list corresponding files
        self.audios_splits_list_files_into = self.extract_splits_list_files(
            data_format="audio_features", splits=['train', 'val'])

        self.video_splits_list_files_into = self.extract_splits_list_files(
            data_format="videos", splits=['train', 'val'])
        self.rawframes_splits_list_files_into = self.extract_splits_list_files(
            data_format="rawframes", splits=['train', 'val'])

    def get_modality_name(self):
        """ Get all supports modalities """
        return ["rgb", "flow", "audio"]

    def rename_classes(self, mode):
        """ Rename classes by replacing whitespace to  'Underscore' """
        video_format_path_key = self.set_modality_path_key_format(
            modality_name="video")
        videos_root__path = self.splits_info[mode][video_format_path_key]
        videos_dirs_name = [
            dir_name for dir_name in os.listdir(videos_root__path)
            if os.path.isdir(os.path.join(videos_root__path, dir_name))
        ]

        new_videos_dirs_name = [
            dir_name.replace(" ", "_") for dir_name in videos_dirs_name
        ]

        videos_dirs_path = [
            os.path.join(videos_root__path, dir_name)
            for dir_name in videos_dirs_name
        ]
        new_videos_dirs_path = [
            os.path.join(videos_root__path, dir_name)
            for dir_name in new_videos_dirs_name
        ]
        for i, _ in enumerate(videos_dirs_path):
            os.rename(videos_dirs_path[i], new_videos_dirs_path[i])

    def get_modality_data_path(self, mode, modality_name):
        """ Obtain the path for the modality data in specific mode """

        modality_key = self.set_modality_path_key_format(
            modality_name=modality_name)

        return self.splits_info[mode][modality_key]

    def extract_videos_rgb_flow_audio(self, mode="train"):
        """ Extract rgb, optical flow, and audio from videos """
        video_data_path = self.get_modality_data_path(mode=mode,
                                                      modality_name="video")
        src_mode_videos_dir = video_data_path

        rgb_out__path = self.get_modality_data_path(mode=mode,
                                                    modality_name="rgb")
        flow_our__path = self.get_modality_data_path(mode=mode,
                                                     modality_name="flow")
        audio_out__path = self.get_modality_data_path(mode=mode,
                                                      modality_name="audio")
        audio_feature__path = self.get_modality_data_path(
            mode=mode, modality_name="audio_feature")

        # define the modalities extractor
        if not self._exists(rgb_out__path):
            vdf_extractor = frames_extraction_tools.VideoFramesExtractor(
                video_src_dir=src_mode_videos_dir,
                dir_level=2,
                num_worker=8,
                video_ext="mp4",
                mixed_ext=False)
        if not self._exists(audio_out__path) \
            or not self._exists(audio_feature__path):
            vda_extractor = audio_extraction_tools.VideoAudioExtractor(
                video_src_dir=src_mode_videos_dir,
                dir_level=2,
                num_worker=8,
                video_ext="mp4",
                mixed_ext=False)

        if torch.cuda.is_available():
            if not self._exists(rgb_out__path):
                logging.info(
                    "Extracting frames by GPU from videos in %s to %s.",
                    src_mode_videos_dir, rgb_out__path)
                vdf_extractor.build_frames_gpu(rgb_out__path,
                                               flow_our__path,
                                               new_short=1,
                                               new_width=0,
                                               new_height=0)
        else:
            if not self._exists(rgb_out__path):
                logging.info(
                    "Extracting frames by CPU from videos in %s to %s.",
                    src_mode_videos_dir, rgb_out__path)
                vdf_extractor.build_frames_cpu(to_dir=rgb_out__path)

        if not self._exists(audio_out__path):
            logging.info("Extracting audios by CPU from videos in %s to %s.",
                         src_mode_videos_dir, audio_out__path)
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

    def extract_splits_list_files(self, data_format, splits):
        """ Extract and generate the split information of current mode/phase """
        output_format = "json"
        out_path = self.mm_data_info["data_path"]

        # obtained a dict that contains the required data splits' file path
        #   it can be full data or tiny data
        required_anno_files = obtain_required_anno_files(self.splits_info)
        data_splits_file_info = required_anno_files
        gen_annots_op = modality_data_anntation_tools.GenerateMDataAnnotation(
            data_src_dir=self.mm_data_info["data_path"],
            data_annos_files_info=data_splits_file_info,
            dataset_name=self.data_name,
            data_format=data_format,  # 'rawframes', 'videos', 'audio_features'
            rgb_prefix="img_",  # prefix of rgb frames
            flow_x_prefix="x_",  # prefix of flow x frames
            flow_y_prefix="y_",  # prefix of flow y frames
            out_path=out_path,
            output_format=output_format)

        target_list_regu = f'_{data_format}.{output_format}'
        if not self._file_exists(tg_file_name=target_list_regu,
                                 search_path=out_path,
                                 is_partial_name=True):
            logging.info("Extracting annotation list for %s. ", data_format)

            gen_annots_op.read_data_splits_csv_info()

            for split_name in splits:
                gen_annots_op.generate_data_splits_info_file(
                    split_name=split_name)

        # obtain the extracted files path
        generated_list_files_info = {}
        for split_name in splits:
            generated_list_files_info[
                split_name] = gen_annots_op.get_anno_file_path(split_name)

        return generated_list_files_info

    def correct_current_config(self, loaded_plato_config, mode, modality_name):
        """Correct the loaded configuration settings based on on-hand data information."""

        # 1.1. convert plato config to dict type
        loaded_config = data_utils.config_to_dict(loaded_plato_config)
        # 1.2. convert the list to tuple
        loaded_config = data_utils.dict_list2tuple(loaded_config)

        # 2. using the obtained annotation file replace the user set ones
        #   in the configuration file
        #   The main reason is that the obtained path here is the full path
        cur_rawframes_anno_file_path = self.rawframes_splits_list_files_into[
            mode]
        cur_rawframes_data_path = self.get_modality_data_path(
            mode=mode, modality_name="rgb")
        cur_videos_anno_file_path = self.video_splits_list_files_into[mode]
        cur_video_data_path = self.get_modality_data_path(
            mode=mode, modality_name="video")
        cur_audio_feas_anno_file_path = self.audios_splits_list_files_into[
            mode]
        cur_audio_feas_data_path = self.get_modality_data_path(
            mode=mode, modality_name="audio_feature")

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
            "audio_feature": audio_feature_mode_config["ann_file"]
        }

        kinetics_mode_dataset = KineticsDataset(
            multimodal_data_holder=multi_modal_mode_data,
            phase="train",
            phase_info=multi_modal_mode_info,
            modality_sampler=modality_sampler)

        return kinetics_mode_dataset

    def get_train_set(self, modality_sampler=None):
        """ Obtain the trainset for multimodal data. """
        kinetics_train_dataset = self.get_phase_dataset(
            phase="train", modality_sampler=modality_sampler)

        return kinetics_train_dataset

    def get_test_set(self, modality_sampler=None):
        """ Obtain the testset for multimodal data.

            Note, in the kinetics dataset, there is no testset in which
             samples contain the groundtruth label.
             Thus, we utilize the validation set directly.
        """
        kinetics_val_dataset = self.get_phase_dataset(
            phase="val", modality_sampler=modality_sampler)

        return kinetics_val_dataset

    def get_class_label_mapper(self):
        """ Obtain the mapper used to map the text to integer. """
        textclass_integer_mapper = defaultdict(list)
        # obtain the classes from the trainset
        train_anno_list_path = self.rawframes_splits_list_files_into["train"]
        train_anno_list = data_utils.read_anno_file(train_anno_list_path)
        # [{"frame_dir": "clay_pottery_making/---0dWlqevI_000019_000029",
        #   "total_frames": 300, "label": [0]}
        for item in train_anno_list:
            textclass = item["frame_dir"].split("/")[0]
            integar_label = item["frame_dir"]["label"][0]

            textclass_integer_mapper[textclass].append(integar_label)

        return textclass_integer_mapper

    def classes(self):
        """ The classes of the dataset. """

        # obtain the classes from the trainset
        train_anno_list_path = self.rawframes_splits_list_files_into["train"]
        train_anno_list = data_utils.read_anno_file(train_anno_list_path)

        integer_labels = [
            anno_elem["label"][0] for anno_elem in train_anno_list
        ]
        integer_classes = list(set(integer_labels))
        integer_classes.sort()

        return integer_classes
