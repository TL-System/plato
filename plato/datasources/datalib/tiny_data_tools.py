"""
This includes tools for creating the processing tiny dataset.

The tiny dataset is sampled from the whole dataset.

"""

import os
import collections
import logging

import pandas as pd
import numpy as np


def create_tiny_kinetics_anno(kinetics_annotation_files_info, num_samples,
                              random_seed):
    """ Creating the annotation files for tiny kinetics.

        Args:
            kinetics_annotation_files_info (dict): a dict contains the annotation
             files for different splits. e.g., {"train": xxx, "val": xxx}.
            num_fist_videos (int): the number of samples utilized to create
             tiny dataset.
            random_seed (int): the random seed for sampling samples.
    """
    train_anno_file_path = kinetics_annotation_files_info["train"]
    test_anno_file_path = kinetics_annotation_files_info["test"]
    val_anno_file_path = kinetics_annotation_files_info["val"]
    np.random.seed(random_seed)

    train_anno_df = pd.read_csv(train_anno_file_path)
    train_selected_samples_df = train_anno_df.iloc[:num_samples]

    train_selected_classes = train_selected_samples_df["label"].tolist()

    # select from test/val anno files based on the train classes
    def select_anchored_samples(src_anno_df, anchor_classes):
        selected_df = None
        counted_classes = collections.Counter(anchor_classes)
        for cls in list(counted_classes.keys()):
            num_cls = counted_classes[cls]
            cls_anno_df = src_anno_df[src_anno_df["label"] == cls]
            len_cls_df = len(cls_anno_df)
            selected_samples_idx = np.random.choice(list(range(len_cls_df)),
                                                    size=num_cls)
            cls_selected_df = cls_anno_df.iloc[selected_samples_idx]
            if selected_df is None:
                selected_df = cls_selected_df
            else:
                selected_df = pd.concat([selected_df, cls_selected_df])
        return selected_df

    def save_anno_df(anno_file_path, selected_anno_df):
        anno_file_base_dir = os.path.dirname(anno_file_path)
        anno_file_base_name, ext_type = os.path.basename(anno_file_path).split(
            ".")
        tiny_anno_file_path = os.path.join(
            anno_file_base_dir, anno_file_base_name + "_tiny." + ext_type)

        if os.path.exists(tiny_anno_file_path):
            logging.info(
                "Annotation file for tiny data exists: %s, Using it directly",
                tiny_anno_file_path)
        else:
            selected_anno_df.to_csv(path_or_buf=tiny_anno_file_path,
                                    index=False)

    test_anno_df = pd.read_csv(test_anno_file_path)
    test_selected_df = test_anno_df.iloc[:num_samples]

    val_anno_df = pd.read_csv(val_anno_file_path)
    val_selected_df = select_anchored_samples(
        src_anno_df=val_anno_df, anchor_classes=train_selected_classes)

    save_anno_df(anno_file_path=train_anno_file_path,
                 selected_anno_df=train_selected_samples_df)
    save_anno_df(anno_file_path=test_anno_file_path,
                 selected_anno_df=test_selected_df)
    save_anno_df(anno_file_path=val_anno_file_path,
                 selected_anno_df=val_selected_df)
