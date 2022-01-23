"""
The Flickr30K Entities dataset.

The data structure and setting follow:
 "http://bryanplummer.com/Flickr30kEntities/".

We utilize the official splits that contain:
 - train: 29783 images,
 - val: 1000 images,
 - test: 1000 images

The file structure of this dataset is:
 - Images (jpg): the raw images
 - Annotations (xml): the bounding boxes
 - Sentence (txt): captions of the image

The data structure under the 'data/' is:
├── Flickr30KEntities           # root dir of Flickr30K Entities dataset
│   ├── Flickr30KEntitiesRaw    # Raw images/annotations and the official splits
│   ├── train     # data dir for the train phase
│   │   └── train_Annotations
│   │   └── train_Images
│   │   └── train_Sentences
│   └── test
│   └── val


Detailed loaded sample structure:

    One sample is presented as the dict type:
    - rgb: the image data.
    - text:
        - caption : a nested list, such as
            [['The woman is applying mascara while looking in the mirror.']],
        - caption_phrases: a nested list, each item is the list contains
            the phrases of the caption, such as:
            [['Military personnel'], ['greenish gray uniforms'], ['matching hats']]
    - box:
        - caption_phrase_bboxs: a 2-depth nested list, each item is a list that
            contains boxes of the corresponding phrase, such as:
            [[[295, 130, 366, 244], [209, 123, 300, 246], [347, 1, 439, 236]],
                [[0, 21, 377, 220]], [[0, 209, 214, 332]]]
    - target:
        - caption_phrases_cate: a nested list, each item is a string that
            presents the categories of the phrase, such as:
            [['people'], ['bodyparts'], ['other']].

        - caption_phrases_cate_id: a list, each item is a int that shows
            the integar/str of the phrase, such as:
            ['121973', '121976', '121975']

    One batch of samples is presented as a list,
        For example, the corresponding caption_phrase_bboxs in one batch is:
[
    [[[295, 130, 366, 244], [209, 123, 300, 246], [347, 1, 439, 236]], [[0, 21, 377, 220]],
        [[0, 209, 214, 332]]], - batch-1
    [[[90, 68, 325, 374]], [[118, 64, 192, 128]]], - batch-1
    [[[1, 0, 148, 451]], [[153, 148, 400, 413]], [[374, 320, 450, 440]]], - batch-1
]
"""

import json
import logging
import os

import torch
import skimage.io as io
import cv2

from plato.config import Config
from plato.datasources import multimodal_base
from plato.datasources.multimodal_base import TextData, BoxData, TargetData
from plato.datasources.datalib import data_utils
from plato.datasources.datalib import flickr30kE_utils


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


class Flickr30KEDataset(multimodal_base.MultiModalDataset):
    """ Prepare the Flickr30K Entities dataset."""
    def __init__(self,
                 dataset_info,
                 phase,
                 phase_info,
                 data_types,
                 modality_sampler=None,
                 transform_image_dec_func=None,
                 transform_text_func=None):
        super().__init__()

        self.phase = phase
        self.phase_multimodal_data_record = dataset_info
        self.phase_info = phase_info
        self.data_types = data_types
        self.transform_image_dec_func = transform_image_dec_func
        self.transform_text_func = transform_text_func

        self.phase_samples_name = list(
            self.phase_multimodal_data_record.keys())

        self.supported_modalities = ["rgb", "text"]

        # default utilizing the full modalities
        if modality_sampler is None:
            self.modality_sampler = self.supported_modalities
        else:
            self.modality_sampler = modality_sampler

    def __len__(self):
        return len(self.phase_multimodal_data_record)

    def get_sample_image_data(self, image_id):
        """ Get one image data as the sample """
        # get the image data
        image_phase_path = self.phase_info[self.data_types[0]]["path"]
        image_phase_format = self.phase_info[self.data_types[0]]["format"]

        image_data = io.imread(
            os.path.join(image_phase_path,
                         str(image_id) + image_phase_format))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        return image_data

    def extract_sample_anno_data(self, image_anno_sent):
        """ Extract the annotation. """
        sentence = image_anno_sent["sentence"]  # a string
        sentence_phrases = image_anno_sent["sentence_phrases"]  # a list
        sentence_phrases_type = image_anno_sent[
            "sentence_phrases_type"]  # a nested list
        sentence_phrases_id = image_anno_sent["sentence_phrases_id"]  # a list
        sentence_phrases_boxes = image_anno_sent[
            "sentence_phrases_boxes"]  # a nested list

        return sentence, sentence_phrases, sentence_phrases_type, \
                sentence_phrases_id, sentence_phrases_boxes

    def get_one_multimodal_sample(self, sample_idx):
        """ Obtain one sample from the Flickr30K Entities dataset. """
        samle_retrieval_name = self.phase_samples_name[sample_idx]
        image_file_name = os.path.basename(samle_retrieval_name)
        image_id = os.path.splitext(image_file_name)[0]

        image_data = self.get_sample_image_data(image_id)

        image_anno_sent = self.phase_multimodal_data_record[
            samle_retrieval_name]

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
            caption_phrase_bboxs = flickr30kE_utils.phrase_boxes_alignment(
                flatten_caption_phrase_bboxs, sentence_phrases_boxes)

        else:
            caption_phrase_bboxs = sentence_phrases_boxes

        if self.transform_text_func is not None:
            caption_phrases = self.transform_text_func(caption_phrases)

        text_data = TextData(caption=caption, caption_phrases=caption_phrases)
        box_data = BoxData(caption_phrase_bboxs=caption_phrase_bboxs)
        taget_data = TargetData(
            caption_phrases_cate=caption_phrases_cate,
            caption_phrases_cate_id=caption_phrases_cate_id)

        return {
            "rgb": image_data,
            "text": text_data,
            "box": box_data,
            "target": taget_data
        }


class DataSource(multimodal_base.MultiModalDataSource):
    """The Flickr30K Entities dataset."""
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

        # extract the data information and structure
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

        # generate path/type information for splits
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

        # distribution data to splits
        self.create_splits_data()

        # generate the splits information txt for further utilization
        flickr30kE_utils.integrate_data_to_json(splits_info=self.splits_info,
                                                mm_data_info=self.mm_data_info,
                                                data_types=self.data_types,
                                                split_wise=True,
                                                globally=True)

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
        dataset = Flickr30KEDataset(dataset_info=phase_data_info,
                                    phase_info=phase_split_info,
                                    data_types=self.data_types,
                                    phase=phase,
                                    modality_sampler=modality_sampler)
        return dataset

    def get_train_set(self, modality_sampler=None):
        """ Obtains the training dataset. """
        phase = "train"

        self.trainset = self.get_phase_dataset(phase, modality_sampler)
        return self.trainset

    def get_test_set(self, modality_sampler=None):
        """ Obtains the validation dataset. """
        phase = "test"

        self.testset = self.get_phase_dataset(phase, modality_sampler)
        return self.testset
