"""
Useful tools for processing the data

"""
import shutil

import numpy as np


def dict_list2tuple(dict_obj):
    """ Convert all list element in the dict to tuple """
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            for inner_key, inner_v in value.items():
                if isinstance(inner_v, list):
                    dict_obj[key][inner_key] = tuple(inner_v)
        else:
            if isinstance(value, list):
                dict_obj[key] = tuple(value)
                for idx, item in enumerate(value):
                    item = value[idx]
                    if isinstance(item, dict):
                        value[idx] = dict_list2tuple(item)

    return dict_obj


def phrase_boxes_alignment(flatten_boxes, ori_phrases_boxes):
    """ Align the phase and its corresponding boxes """
    phrases_boxes = list()

    ori_pb_boxes_count = list()
    for ph_boxes in ori_phrases_boxes:
        ori_pb_boxes_count.append(len(ph_boxes))

    strat_point = 0
    for pb_boxes_num in ori_pb_boxes_count:
        sub_boxes = list()
        for i in range(strat_point, strat_point + pb_boxes_num):
            sub_boxes.append(flatten_boxes[i])

        strat_point += pb_boxes_num
        phrases_boxes.append(sub_boxes)

    pb_boxes_count = list()
    for ph_boxes in phrases_boxes:
        pb_boxes_count.append(len(ph_boxes))

    assert pb_boxes_count == ori_pb_boxes_count

    return phrases_boxes


def list_inorder(listed_files, flag_str):
    """" List the files in order based on the file name """
    filtered_listed_files = [fn for fn in listed_files if flag_str in fn]
    listed_files = sorted(filtered_listed_files,
                          key=lambda x: x.strip().split(".")[0])
    return listed_files


def copy_files(src_files, dst_dir):
    """ copy files from src to dst """
    for file in src_files:
        shutil.copy(file, dst_dir)


def union_shuffled_lists(src_lists):
    """ shuffle the lists """
    for i in range(1, len(src_lists)):
        assert len(src_lists[i]) == len(src_lists[i - 1])
    processed = np.random.permutation(len(src_lists[0]))

    return [np.array(ele)[processed] for ele in src_lists]
