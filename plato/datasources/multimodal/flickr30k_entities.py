"""
The Flickr30K Entities dataset following "http://bryanplummer.com/Flickr30kEntities/"

"""

import json
import logging
import os
from collections import namedtuple

import torch
import skimage.io as io
import cv2

from plato.config import Config
from plato.datasources.multimodal import multimodal_base
from plato.datasources.datalib import data_utils
from plato.datasources.datalib.flicker30k_utils import flickr30k_utils

DataAnnos = namedtuple('annos', [
    'caption', 'caption_phrases', 'caption_phrase_bboxs',
    'caption_phrases_cate', 'caption_phrases_cate_id'
])


def collate_fn(batch):
    """ The construction of the loaded batch of data

    Args:
        batch (list): [a list in which each element contains the data for one task,
                        assert len(batch) == number of tasks,
                        assert len(batch[i]) == 6]

    Returns:
        [batch]: [return the original batch of data directly]
    """
    return batch


class Flickr30KEDataset(torch.utils.data.Dataset):
    """Prepares the Flickr30K Entities dataset."""
    def __init__(self,
                 dataset,
                 splits_info,
                 data_types,
                 phase,
                 transform_image_dec_func=None,
                 transform_text_func=None):
        self.phase = phase
        self.phase_data = dataset
        self.splits_info = splits_info
        self.data_types = data_types
        self.transform_image_dec_func = transform_image_dec_func
        self.transform_text_func = transform_text_func

    def __len__(self):
        return len(self.phase_data)

    def get_sample_image_data(self, phase, image_id):
        """ Get one image data as the sample """
        # get the image data
        image_phase_path = self.splits_info[phase][self.data_types[0]]["path"]
        image_phase_format = self.splits_info[phase][
            self.data_types[0]]["format"]

        image_data = io.imread(
            os.path.join(image_phase_path,
                         str(image_id) + image_phase_format))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        return image_data

    def extract_sample_anno_data(self, image_anno_sent):
        """ Extract the annotation """
        sentence = image_anno_sent["sentence"]  # a string
        sentence_phrases = image_anno_sent["sentence_phrases"]  # a list
        sentence_phrases_type = image_anno_sent[
            "sentence_phrases_type"]  # a nested list
        sentence_phrases_id = image_anno_sent["sentence_phrases_id"]  # a list
        sentence_phrases_boxes = image_anno_sent[
            "sentence_phrases_boxes"]  # a nested list

        return sentence, sentence_phrases, sentence_phrases_type, \
                sentence_phrases_id, sentence_phrases_boxes

    def __getitem__(self, sample_idx):
        samle_retrieval_name = self.phase_samples_name[sample_idx]
        image_id = samle_retrieval_name.split(".")[0]
        sample_name = image_id

        image_data = self.get_sample_image_data(self.phase, image_id)

        ori_image_data = image_data.copy()

        image_anno_sent = self.phase_data[samle_retrieval_name]

        sentence, sentence_phrases, \
            sentence_phrases_type, sentence_phrases_id, \
            sentence_phrases_boxes = self.extract_sample_anno_data(image_anno_sent)

        caption = sentence if any(isinstance(iter_i, list) for iter_i in sentence) \
                                            else [[sentence]]
        flatten_caption_phrase_bboxs = [
            box for boxes in sentence_phrases_boxes for box in boxes
        ]
        # ['The woman', 'mascara', 'the mirror']
        caption_phrases = [[phrase] for phrase in sentence_phrases]
        caption_phrases_cate = sentence_phrases_type
        caption_phrases_cate_id = sentence_phrases_id

        if self.transform_image_dec_func is not None:

            transformed = self.transform_image_dec_func(
                image=image_data,
                bboxes=flatten_caption_phrase_bboxs,
                category_ids=range(len(flatten_caption_phrase_bboxs)))

            image_data = transformed["image"]
            image_data = torch.from_numpy(image_data)
            flatten_caption_phrase_bboxs = transformed["bboxes"]
            caption_phrase_bboxs = flickr30k_utils.phrase_boxes_alignment(
                flatten_caption_phrase_bboxs, sentence_phrases_boxes)

        if self.transform_text_func is not None:
            caption_phrases = self.transform_text_func(caption_phrases)

        sample_annos = DataAnnos(
            caption=caption,
            caption_phrases=caption_phrases,
            caption_phrase_bboxs=caption_phrase_bboxs,
            caption_phrases_cate=caption_phrases_cate,
            caption_phrases_cate_id=caption_phrases_cate_id)

        return sample_name, ori_image_data, image_data, sample_annos


class DataSource(multimodal_base.MultiModalDataSource):
    """The ReferItGame dataset."""
    def __init__(self):
        super().__init__()

        self.data_name = Config().data.dataname

        self.modality_names = ["image", "text"]

        _path = Config().data.data_path
        self._data_path_process(data_path=_path, base_data_name=self.data_name)

        raw_data_name = self.data_name + "Raw"
        base_data_path = self.mm_data_info["base_data_dir_path"]

        download_file_id = Config().data.download_file_id

        self._download_google_driver_arrange_data(
            download_file_id=download_file_id,
            extract_download_file_name=raw_data_name,
            put_data_dir=base_data_path,
        )

        # define the path of different data source,
        #   the annotation is .xml, the sentence is in .txt
        self.raw_data_types = ["Flickr30k_images", "Annotations", "Sentences"]
        self.raw_data_file_format = [".jpg", ".xml", ".txt"]
        self.data_types = ["Images", "Annotations", "Sentences"]

        for raw_type_idx, raw_type in enumerate(self.raw_data_types):
            raw_file_format = self.raw_data_file_format[raw_type_idx]
            data_type = self.data_types[raw_type_idx]

            raw_type_path = os.path.join(base_data_path, raw_data_name,
                                         raw_type)

            self.mm_data_info[data_type] = dict()
            self.mm_data_info[data_type]["path"] = raw_type_path
            self.mm_data_info[data_type]["format"] = raw_file_format
            self.mm_data_info[data_type]["num_samples"] = len(
                os.listdir(raw_type_path))

        for split_type in list(self.splits_info.keys()):
            self.splits_info[split_type]["split_file"] = os.path.join(
                base_data_path, raw_data_name, split_type + ".txt")
            split_path = self.splits_info[split_type]["path"]
            for dt_type_idx, dt_type in enumerate(self.data_types):
                dt_type_format = self.raw_data_file_format[dt_type_idx]

                self.splits_info[split_type][dt_type] = dict()
                self.splits_info[split_type][dt_type]["path"] = os.path.join(
                    split_path, ("{}_{}").format(split_type, dt_type))
                self.splits_info[split_type][dt_type][
                    "format"] = dt_type_format

        print(self.mm_data_info)

    def create_splits_data(self):
        """ Create datasets for different splits """
        # saveing the images and entities to the corresponding directory
        for split_type in list(self.splits_info.keys()):
            logging.info("Creating split %s data..........", split_type)
            # obtain the split data information
            # 0. getting the data
            split_info_file = self.splits_info[split_type]["split_file"]
            with open(split_info_file, "r") as loaded_file:
                split_data_samples = [
                    sample_id.split("\n")[0]
                    for sample_id in loaded_file.readlines()
                ]
            self.splits_info[split_type]["num_samples"] = len(
                split_data_samples)

            # 1. create directory for the splited data if necessary
            for dt_type in self.data_types:
                split_dt_type_path = self.splits_info[split_type][dt_type][
                    "path"]

                if not self._exist_judgement(split_dt_type_path):
                    os.makedirs(split_dt_type_path, exist_ok=True)
                else:
                    logging.info("The path %s does exist", split_dt_type_path)
                    continue

                raw_data_type_path = self.mm_data_info[dt_type]["path"]
                raw_data_format = self.mm_data_info[dt_type]["format"]
                split_samples_path = [
                    os.path.join(raw_data_type_path,
                                 sample_id + raw_data_format)
                    for sample_id in split_data_samples
                ]
                # 2. saving the splited data into the target file
                data_utils.copy_files(split_samples_path, split_dt_type_path)

        logging.info(" Done!")

    def get_phase_data(self, phase):
        """ Obtain the data information for the required phrase """
        path = self.splits_info[phase]["path"]
        save_path = os.path.join(path, phase + "_integrated_data.json")
        with open(save_path, 'r') as outfile:
            phase_data = json.load(outfile)
        return phase_data

    def get_train_loader(self, batch_size):
        """ Obtain the train loader """
        phase = "train"
        phase_data = self.get_phase_data(phase)
        self.trainset = Flickr30KEDataset(dataset=phase_data,
                                          splits_info=self.splits_info,
                                          data_types=self.data_types,
                                          phase=phase)
        train_loader = torch.utils.data.DataLoader(dataset=self.trainset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        return train_loader

    def get_test_loader(self, batch_size):
        """ Obtain the test loader """
        phase = "test"
        phase_data = self.get_phase_data(phase)
        self.testset = Flickr30KEDataset(dataset=phase_data,
                                         splits_info=self.splits_info,
                                         data_types=self.data_types,
                                         phase=phase)
        test_loader = torch.utils.data.DataLoader(dataset=self.testset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=collate_fn)
        return test_loader
