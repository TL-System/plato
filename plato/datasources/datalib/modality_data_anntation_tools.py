"""
The class in this file is supported by the mmaction/tools/data/build_file_list


"""

import os
import glob
import json

from mmaction.tools.data.anno_txt2json import lines2dictlist
from mmaction.tools.data.parse_file_list import parse_directory

from plato.datasources.datalib.parse_datasets import build_list, obtain_data_splits_info


class GenerateMDataAnnotation(object):
    """ Generate the annotation file for the existing data modality """
    def __init__(
        self,
        data_src_dir,
        data_annos_files_info,  # a dict that contains the data splits' file path
        data_format,  # 'rawframes', 'videos'
        out_path,
        dataset_name,
        data_dir_level=2,
        rgb_prefix="img_'",  # prefix of rgb frames
        flow_x_prefix="flow_x_",  # prefix of flow x frames
        flow_y_prefix="flow_y_",  # prefix of flow y frames
        # shuffle=False,  # whether to shuffle the file list
        output_format="json"):  # txt or json

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

        frame_info = None
        if data_format == 'rawframes':
            frame_info = parse_directory(data_src_dir,
                                         rgb_prefix=rgb_prefix,
                                         flow_x_prefix=flow_x_prefix,
                                         flow_y_prefix=flow_y_prefix,
                                         level=data_dir_level)
        elif data_format == 'videos':
            if data_dir_level == 1:
                # search for one-level directory
                video_list = glob.glob(os.path.join(data_src_dir, '*'))
            elif data_dir_level == 2:
                # search for two-level directory
                video_list = glob.glob(os.path.join(data_src_dir, '*', '*'))
            else:
                raise ValueError(
                    f'level must be 1 or 2, but got {self.data_dir_level}')
            frame_info = {}
            for video in video_list:
                video_path = os.path.relpath(video, data_src_dir)
                # video_id: (video_relative_path, -1, -1)
                frame_info[os.path.splitext(video_path)[0]] = (video_path, -1,
                                                               -1)
        else:
            raise NotImplementedError(
                'only rawframes and videos are supported')
        self.frame_info = frame_info

    def generate_data_splits_info_file(self, data_name="kinetics700"):
        """ Generate the data split information and write the info to file """
        data_splits_info = obtain_data_splits_info(
            data_annos_files_info=self.data_annos_files_info,
            data_fir_level=2,
            data_name=data_name)

        for split_name in list(data_splits_info.keys()):
            split_info = data_splits_info[split_name]
            # (rgb_list, flow_list)
            split_built_list = build_list(split=split_info,
                                          frame_info=self.frame_info,
                                          shuffle=False)
            filename = f'{self.dataset_name}_{split_name}_list_{self.data_format}.txt'

            if self.output_format == 'txt':
                with open(os.path.join(self.annotations_out_path, filename),
                          'w') as anno_file:
                    anno_file.writelines(split_built_list[0])
            elif self.output_format == 'json':
                data_list = lines2dictlist(split_built_list[0],
                                           self.data_format)
                filename = filename.replace('.txt', '.json')
                with open(os.path.join(self.annotations_out_path, filename),
                          'w') as anno_file:
                    json.dump(data_list, anno_file)
