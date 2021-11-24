"""
This is the interface for the ReferitGame dataset that includes refer+, refer, referg sub-datasets.

http://tamaraberg.com/referitgame/
"""

import logging
import os

import collections

import torch

import cv2

from plato.config import Config
from plato.datasources.multimodal import multimodal_base
from plato.datasources.datalib.refer_utils import referitgame_utils

DataAnnos = collections.namedtuple('annos', [
    'caption', 'caption_phrases', 'caption_phrase_bboxs',
    'caption_phrases_cate', 'caption_phrases_cate_id'
])

SplitedDatasets = collections.namedtuple('SplitedDatasets', [
    'train_ref_ids', 'val_ref_ids', 'test_ref_ids', 'testA_ref_ids',
    'testB_ref_ids', 'testC_ref_ids'
])


def collate_fn(batch):
    """[The construction of the loaded batch of data]

    Args:
        batch ([list]): [a list in which each element contains the data for one task,
                        assert len(batch) == number of tasks,
                        assert len(batch[i]) == 6 that is the output of \
                            create_task_examples_data function]

    Returns:
        [batch]: [return the original batch of data directly]
    """
    return batch


class ReferItGameDataset(torch.utils.data.Dataset):
    """Prepares the ReferItGame dataset for use in the model."""
    def __init__(self,
                 dataset,
                 base_refer_data,
                 transform_image_dec_func=None,
                 transform_text_func=None):
        self.phase_data = dataset
        self.base_refer_data = base_refer_data
        self.transform_image_dec_func = transform_image_dec_func
        self.transform_text_func = transform_text_func

    def __len__(self):
        return len(self.phase_data)

    def __getitem__(self, sample_idx):
        [
            image_id, _, caption, caption_phrases, caption_phrase_bboxs,
            caption_phrases_cate, caption_phrases_cate_id
        ] = self.phase_data[sample_idx]

        sample_name = image_id
        image_data = self.base_refer_data.loadImgsData(image_id)[0]

        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        ori_image_data = image_data.copy()

        caption = caption if any(isinstance(boxes_i, list) for boxes_i in caption) \
                                            else [caption]
        caption_phrase_bboxs = caption_phrase_bboxs if any(isinstance(boxes_i, list) \
                                                for boxes_i in caption_phrase_bboxs) \
                                                    else [caption_phrase_bboxs]
        caption_phrases = caption_phrases if any(isinstance(boxes_i, list) \
                                        for boxes_i in caption_phrases) \
                                            else [caption_phrases]
        caption_phrases_cate = caption_phrases_cate if any(isinstance(boxes_i, list) \
                                        for boxes_i in caption_phrases_cate) \
                                            else [[caption_phrases_cate]]
        caption_phrases_cate_id = caption_phrases_cate_id \
                                            if isinstance(caption_phrases_cate_id, list) \
                                            else [caption_phrases_cate_id]

        assert len(caption_phrase_bboxs) == len(caption_phrases)
        if self.transform_image_dec_func is not None:

            transformed = self.transform_image_dec_func(
                image=image_data,
                bboxes=caption_phrase_bboxs,
                category_ids=caption_phrases_cate_id)

            image_data = transformed["image"]
            image_data = torch.from_numpy(image_data)
            caption_phrase_bboxs = transformed["bboxes"]

        if self.transform_text_func is not None:
            caption_phrases = self.transform_text_func(caption_phrases)

        caption_phrase_bboxs = [caption_phrase_bboxs
                                ]  # convert to the standard structure

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

        self.split_configs = ["refcoco", "refcoco+", "refcocog"]

        self.data_name = Config().data.dataname
        self.data_source = Config().data.datasource
        self.split_config = Config().data.split_config
        self.split_name = Config().data.split_name
        if self.split_config not in self.split_configs:
            info_msg = (
                "{} does not exsit in the official configs {}.....").format(
                    self.split_config, self.split_configs)
            logging.info(info_msg)

        self.modality_names = ["image", "text"]

        _path = Config().data.data_path
        self._data_path_process(data_path=_path, base_data_name=self.data_name)
        base_data_path = self.mm_data_info["base_data_dir_path"]

        # the source data is required
        source_data_path = os.path.join(_path, self.data_source)
        if not self._exist_judgement(source_data_path):
            info_msg = (
                "The source data {} must be downloaded first to the directory {} "
            ).format(self.data_source, self.split_configs)
            logging.info(info_msg)
            exit()

        # download the public official code and the required config
        download_split_url = Config(
        ).data.download_splits_base_url + self.split_config + ".zip"
        for dd_url in [download_split_url]:
            self._download_arrange_data(download_url_address=dd_url,
                                        put_data_dir=base_data_path)

        # raw coco images path
        coco_raw_imgs_path = os.path.join(source_data_path, "COCO2017Raw",
                                          "train2017")
        if self._exist_judgement(coco_raw_imgs_path):
            logging.info(
                "Successfully connecting the source COCO2017 images data from the path %s",
                coco_raw_imgs_path)
        self._dataset_refer = referitgame_utils.REFER(
            data_root=base_data_path,
            image_dataroot=coco_raw_imgs_path,
            dataset=self.split_config,
            splitBy=self.split_name)  # default is unc or google

        self._splited_referids_holder = dict()
        self._connect_to_splits()

    def _connect_to_splits(self):
        split_types = SplitedDatasets._fields
        for split_type in split_types:
            formatted_split_type = split_type.split("_", maxsplit=1)[0]
            self._splited_referids_holder[
                formatted_split_type] = self._dataset_refer.getRefIds(
                    split=formatted_split_type)

    def get_phase_data(self, phase):
        """ Get phrases from the raw data """
        mode_refer_ids = self._splited_referids_holder[phase]

        mode_elements_holder = dict()
        mode_flatten_emelemts = list()

        for refer_id in mode_refer_ids:

            ref = self._dataset_refer.loadRefs(refer_id)[0]
            image_id = ref['image_id']
            image_file_path = self._dataset_refer.loadImgspath(image_id)
            caption_phrases_cate = self._dataset_refer.Cats[ref['category_id']]
            caption_phrases_cate_id = ref['category_id']

            mode_elements_holder[refer_id] = dict()
            mode_elements_holder[refer_id]["image_id"] = image_id
            mode_elements_holder[refer_id]["image_file_path"] = image_file_path

            mode_elements_holder[refer_id]["sentences"] = list()
            for send in ref["sentences"]:
                caption = send["tokens"]
                caption_phrase = send["tokens"]

                # images_data = dt_refer.loadImgData(image_id) # a list
                caption_phrase_bboxs = self._dataset_refer.getRefBox(
                    ref['ref_id'])  # [x, y, w, h]
                # convert to [xmin, ymin, xmax, ymax]
                caption_phrase_bboxs = [
                    caption_phrase_bboxs[0], caption_phrase_bboxs[1],
                    caption_phrase_bboxs[0] + caption_phrase_bboxs[2],
                    caption_phrase_bboxs[1] + caption_phrase_bboxs[3]
                ]

                sent_infos = {
                    "caption": caption,
                    "caption_phrase": caption_phrase,
                    "caption_phrase_bboxs": caption_phrase_bboxs,
                    "caption_phrases_cate": caption_phrases_cate,
                    "caption_phrases_cate_id": caption_phrases_cate_id
                }

                mode_elements_holder[refer_id]["sentences"].append(sent_infos)

                mode_flatten_emelemts.append([
                    image_id, image_file_path, caption, caption_phrase,
                    caption_phrase_bboxs, caption_phrases_cate,
                    caption_phrases_cate_id
                ])

        return mode_elements_holder, mode_flatten_emelemts

    def get_train_loader(self, batch_size):
        """ Get the train loader """
        phase = "train"
        _, mode_flatten_emelemts = self.get_phase_data(phase)
        self.trainset = ReferItGameDataset(dataset=mode_flatten_emelemts,
                                           base_refer_data=self._dataset_refer)
        train_loader = torch.utils.data.DataLoader(dataset=self.trainset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        return train_loader

    def get_test_loader(self, batch_size):
        """ Get the test loader """
        phase = "test"
        _, mode_flatten_emelemts = self.get_phase_data(phase)
        self.testset = ReferItGameDataset(dataset=mode_flatten_emelemts,
                                          base_refer_data=self._dataset_refer)
        test_loader = torch.utils.data.DataLoader(dataset=self.testset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=collate_fn)
        return test_loader
