""" This part of the code heavily depends on the
    tools/data/build_file_lists.py provided by the mmaction

"""

import csv
import random

from mmaction.tools.data.parse_file_list import (
    parse_diving48_splits, parse_hmdb51_split, parse_jester_splits,
    parse_mit_splits, parse_mmit_splits, parse_sthv1_splits,
    parse_sthv2_splits, parse_ucf101_splits)


def build_list(split, frame_info, shuffle=False):
    """Build RGB and Flow file list with a given split.

    Args:
        split (list): Split to be generate file list.

    Returns:
        tuple[list, list]: (rgb_list, flow_list), rgb_list is the
            generated file list for rgb, flow_list is the generated
            file list for flow.
    """
    rgb_list, flow_list = list(), list()
    for item in split:
        if item[0] not in frame_info:
            continue
        elif frame_info[item[0]][1] > 0:
            # rawframes
            rgb_cnt = frame_info[item[0]][1]
            flow_cnt = frame_info[item[0]][2]
            if isinstance(item[1], int):
                rgb_list.append(f'{item[0]} {rgb_cnt} {item[1]}\n')
                flow_list.append(f'{item[0]} {flow_cnt} {item[1]}\n')
            elif isinstance(item[1], list):
                # only for multi-label datasets like mmit
                rgb_list.append(f'{item[0]} {rgb_cnt} ' +
                                ' '.join([str(digit)
                                          for digit in item[1]]) + '\n')
                rgb_list.append(f'{item[0]} {flow_cnt} ' +
                                ' '.join([str(digit)
                                          for digit in item[1]]) + '\n')
            else:
                raise ValueError(
                    'frame_info should be ' +
                    '[`video`(str), `label`(int)|`labels(list[int])`')
        else:
            # videos
            if isinstance(item[1], int):
                rgb_list.append(f'{frame_info[item[0]][0]} {item[1]}\n')
                flow_list.append(f'{frame_info[item[0]][0]} {item[1]}\n')
            elif isinstance(item[1], list):
                # only for multi-label datasets like mmit
                rgb_list.append(f'{frame_info[item[0]][0]} ' +
                                ' '.join([str(digit)
                                          for digit in item[1]]) + '\n')
                flow_list.append(f'{frame_info[item[0]][0]} ' +
                                 ' '.join([str(digit)
                                           for digit in item[1]]) + '\n')
            else:
                raise ValueError(
                    'frame_info should be ' +
                    '[`video`(str), `label`(int)|`labels(list[int])`')
    if shuffle:
        random.shuffle(rgb_list)
        random.shuffle(flow_list)
    return (rgb_list, flow_list)


def parse_kinetics_splits(kinetics_anntation_files_info, level, dataset_name):
    """Parse Kinetics dataset into "train", "val", "test" splits.

    Args:
        kinetics_anntation_files_info (dict): The file path of the original annotation file.
                                        The file should be the "*.csv" provided in the
                                        official website.
                                        For example:
                                            {"train": ""}
        level (int): Directory level of data. 1 for the single-level directory,
            2 for the two-level directory.
        dataset (str): Denotes the version of Kinetics that needs to be parsed,
            choices are "kinetics400", "kinetics600" and "kinetics700".

    Returns:
        list: "train", "val", "test" splits of Kinetics.
    """
    def convert_label(label_str, keep_whitespaces=False):
        """Convert label name to a formal string.

        Remove redundant '"' and convert whitespace to '_'.

        Args:
            label_str (str): String to be converted.
            keep_whitespaces(bool): Whether to keep whitespace. Default: False.

        Returns:
            str: Converted string.
        """
        if not keep_whitespaces:
            return label_str.replace('"', '').replace(' ', '_')
        else:
            return label_str.replace('"', '')

    def line_to_map(line_str, test=False):
        """A function to map line string to video and label.

        Args:
            line_str (str): A single line from Kinetics csv file.
            test (bool): Indicate whether the line comes from test
                annotation file.

        Returns:
            tuple[str, str]: (video, label), video is the video id,
                label is the video label.
        """
        if test:  # x:  ['---v8pgm1eQ', '0', '10', 'test']
            # video = f'{x[0]}_{int(x[1]):06d}_{int(x[2]):06d}'
            # video = f'{x[1]}_{int(float(x[2])):06d}_{int(float(x[3])):06d}'
            video = f'{line_str[0]}_{int(float(line_str[1])):06d}_{int(float(line_str[2])):06d}'
            label = -1  # label unknown
            return video, label
        else:  # ['clay pottery making', '---0dWlqevI', '19', '29', 'train']
            video = f'{line_str[1]}_{int(float(line_str[2])):06d}_{int(float(line_str[3])):06d}'
            if level == 2:
                video = f'{convert_label(line_str[0])}/{video}'
            else:
                assert level == 1
            label = class_mapping[convert_label(line_str[0])]
            return video, label

    assert "train" in list(kinetics_anntation_files_info.keys())

    splits = dict()
    for split_name in ["train", "val", "test"]:
        if split_name not in list(kinetics_anntation_files_info.keys()):
            continue
        split_anno_info = kinetics_anntation_files_info[split_name]

        # obtain the class map information
        if split_name == "train:":
            csv_reader = csv.reader(open(split_anno_info))
            # skip the first line
            next(csv_reader)
            labels_sorted = sorted(
                set([convert_label(row[0]) for row in csv_reader]))
            class_mapping = {label: i for i, label in enumerate(labels_sorted)}

        csv_reader = csv.reader(open(split_anno_info))
        next(csv_reader)
        if split_name == "test:":
            obtained_split_list = [
                line_to_map(x, test=True) for x in csv_reader
            ]
        else:
            obtained_split_list = [line_to_map(x) for x in csv_reader]

        splits[split_name] = obtained_split_list

    return (splits, )


def obtain_data_splits_info(
        data_annos_files_info,  # a dict containing the data original splits' file path
        data_fir_level=2,
        data_name="kinetics700"):
    """ Parse the raw data file to obtain different splits info """
    if data_name == 'ucf101':
        splits = parse_ucf101_splits(data_fir_level)
    elif data_name == 'sthv1':
        splits = parse_sthv1_splits(data_fir_level)
    elif data_name == 'sthv2':
        splits = parse_sthv2_splits(data_fir_level)
    elif data_name == 'mit':
        splits = parse_mit_splits()
    elif data_name == 'mmit':
        splits = parse_mmit_splits()
    elif data_name in ['kinetics400', 'kinetics600', 'kinetics700']:
        kinetics_anntation_files_info = data_annos_files_info
        splits = parse_kinetics_splits(kinetics_anntation_files_info,
                                       data_fir_level, data_name)
    elif data_name == 'hmdb51':
        splits = parse_hmdb51_split(data_fir_level)
    elif data_name == 'jester':
        splits = parse_jester_splits(data_fir_level)
    elif data_name == 'diving48':
        splits = parse_diving48_splits()
    else:
        raise ValueError(
            f"Supported datasets are 'ucf101, sthv1, sthv2', 'jester', "
            f"'mmit', 'mit', 'kinetics400', 'kinetics600', 'kinetics700', but "
            f'got {data_name}')

    return splits
