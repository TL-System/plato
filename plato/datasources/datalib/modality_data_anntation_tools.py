"""
The class in this file is supported by the mmaction/tools/data/build_file_list


"""

import glob
import json
import os

from mmaction.tools.data.anno_txt2json import lines2dictlist
from mmaction.tools.data.parse_file_list import parse_directory

from plato.datasources.datalib.parse_datasets import build_list, obtain_data_splits_info


class GenerateMDataAnnotation(object):
    """Generate the annotation file for the existing data modality"""

    def __init__(
        self,
        data_src_dir,
        data_annos_files_info,  # a dict that contains the data splits' file path
        data_format,  # 'rawframes', 'videos'
        out_path,
        dataset_name,
        data_dir_level=2,
        rgb_prefix="img_'",  # prefix of rgb frames
        flow_x_prefix="flow_x_",  # prefix of flow x frames [flow_x_ or x_]
        flow_y_prefix="flow_y_",  # prefix of flow y frames [flow_y_ or y_]
        # shuffle=False,  # whether to shuffle the file list
        output_format="json",
    ):  # txt or json
        self.data_src_dir = data_src_dir
        self.data_annos_files_info = data_annos_files_info
        self.dataset_name = dataset_name
        self.data_format = data_format
        self.annotations_out_path = out_path
        self.data_dir_level = data_dir_level
        self.rgb_prefix = rgb_prefix
        self.flow_x_prefix = flow_x_prefix
        self.flow_y_prefix = flow_y_prefix

        self.output_format = output_format

        self.data_splits_info = None
        self.frame_info = None

    def read_data_splits_csv_info(self):
        """Get the data splits information from the csv annotation files"""
        self.data_splits_info = obtain_data_splits_info(
            data_annos_files_info=self.data_annos_files_info,
            data_fir_level=2,
            data_name=self.dataset_name,
        )

    def parse_levels_dir(self, data_src_dir):
        data_dir_info = {}
        """ Parse the dir with several levels. """
        if self.data_dir_level == 1:
            # search for one-level directory
            files_list = glob.glob(os.path.join(data_src_dir, "*"))
        elif self.data_dir_level == 2:
            # search for two-level directory
            files_list = glob.glob(os.path.join(data_src_dir, "*", "*"))
        else:
            raise ValueError(f"level must be 1 or 2, but got {self.data_dir_level}")
        for file in files_list:
            file_path = os.path.relpath(file, data_src_dir)
            # for video: video_id: (video_relative_path, -1, -1)
            # for audio: audio_id: (audio_relative_path, -1, -1)
            data_dir_info[os.path.splitext(file_path)[0]] = (file_path, -1, -1)

        return data_dir_info

    def parse_dir_files(self, split):
        """Parse the dir to summary the data information"""
        # The annotations for audio spectrogram features are identical to those of rawframes.

        # data_format = "rawframes" if self.data_format == "audio_features" else self.data_format
        # split_format_data_src_dir = os.path.join(self.data_src_dir, split,
        #                                          data_format)

        split_format_data_src_dir = os.path.join(
            self.data_src_dir, split, self.data_format
        )
        frame_info = None
        if self.data_format == "rawframes":
            frame_info = parse_directory(
                split_format_data_src_dir,
                rgb_prefix=self.rgb_prefix,
                flow_x_prefix=self.flow_x_prefix,
                flow_y_prefix=self.flow_y_prefix,
                level=self.data_dir_level,
            )
        elif self.data_format == "videos":
            frame_info = self.parse_levels_dir(split_format_data_src_dir)
        elif self.data_format in ["audio_features", "audios"]:
            # the audio anno list should be consistent with that of rawframes
            rawframes_src_path = os.path.join(self.data_src_dir, split, "rawframes")
            frame_info = parse_directory(
                rawframes_src_path,
                rgb_prefix=self.rgb_prefix,
                flow_x_prefix=self.flow_x_prefix,
                flow_y_prefix=self.flow_y_prefix,
                level=self.data_dir_level,
            )
        else:
            raise NotImplementedError("only rawframes and videos are supported")
        self.frame_info = frame_info

    def get_anno_file_path(self, split_name):
        """Get the annotation file path"""
        filename = f"{self.dataset_name}_{split_name}_list_{self.data_format}.txt"

        if self.output_format == "json":
            filename = filename.replace(".txt", ".json")

        output_anno_file_path = os.path.join(self.annotations_out_path, filename)

        return output_anno_file_path

    def generate_data_splits_info_file(self, split_name):
        """Generate the data split information and write the info to file"""
        self.parse_dir_files(split_name)

        split_info = self.data_splits_info[split_name]

        # (rgb_list, flow_list)
        split_built_list = build_list(
            split=split_info, frame_info=self.frame_info, shuffle=False
        )

        output_file_path = self.get_anno_file_path(split_name=split_name)

        data_format = (
            "rawframes"
            if self.data_format in ["audio_features", "audios"]
            else self.data_format
        )

        if self.output_format == "txt":
            with open(output_file_path, "w") as anno_file:
                anno_file.writelines(split_built_list[0])
        elif self.output_format == "json":
            data_list = lines2dictlist(split_built_list[0], data_format)
            if self.data_format in ["audios", "audio_features"]:

                def change_title_func(elem):
                    """Using this function to"""
                    # added the filename key with value presenting the
                    #  path of the corresponding video
                    if self.data_format == "audio_features":
                        elem["audio_path"] = elem["frame_dir"] + ".npy"
                    else:
                        elem["audio_path"] = elem["frame_dir"] + ".wav"

                    return elem

                data_list = [change_title_func(elem) for elem in data_list]

            with open(output_file_path, "w") as anno_file:
                json.dump(data_list, anno_file)
