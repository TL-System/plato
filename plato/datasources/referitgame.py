"""

Although the name of this dataset is referitgame, it actually contains four datasets:
 - ReferItGame http://tamaraberg.com/referitgame/.
 Then, refer-based datasets http://vision2.cs.unc.edu/refer/:
 - RefCOCO
 - RefCOCO+
 - RefCOCOg

The 'split_config' needed to be set to support the following datasets:
- referitgame: 130,525 expressions for referring to 96,654 objects in 19,894 images.
                The samples are splited into three subsets.  train/54,127 referring expressions.
                test/5,842, val/60,103 referring expressions.
- refcoco: 142,209 refer expressions for 50,000 objects.
- refcoco+: 141,564 expressions for 49,856 objects.
- refcocog (google):  25,799 images with 49,856 referred objects and expressions.

The output sample structure of this data is consistent with that
 in the flickr30k entities dataset.

"""

import logging

import collections

import torch
import cv2

from plato.config import Config
from plato.datasources import multimodal_base
from plato.datasources.multimodal_base import TextData, BoxData, TargetData
from plato.datasources.datalib.refer_utils import referitgame_utils

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


class ReferItGameDataset(multimodal_base.MultiModalDataset):
    """Prepares the Flickr30K Entities dataset."""

    def __init__(self,
                 dataset_info,
                 phase,
                 phase_info,
                 modality_sampler=None,
                 transform_image_dec_func=None,
                 transform_text_func=None):
        super().__init__()

        self.phase = phase
        self.phase_multimodal_data_record = dataset_info
        self.phase_info = phase_info
        self.transform_image_dec_func = transform_image_dec_func
        self.transform_text_func = transform_text_func

        # The phase data record in referitgame is a list,
        #  each item contains information of one image as
        #  presented in line-258.
        self.phase_samples_name = self.phase_multimodal_data_record

        self.supported_modalities = ["rgb", "text"]

        # Default, utilizing the full modalities
        if modality_sampler is None:
            self.modality_sampler = self.supported_modalities
        else:
            self.modality_sampler = modality_sampler

    def __len__(self):
        return len(self.phase_multimodal_data_record)

    def get_one_multimodal_sample(self, sample_idx):
        [
            image_id, _, caption, caption_phrases, caption_phrase_bboxs,
            caption_phrases_cate, caption_phrases_cate_id
        ] = self.phase_multimodal_data_record[sample_idx]

        _ = image_id
        image_data = self.phase_info.loadImgsData(image_id)[0]

        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        _ = image_data.copy()

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
    """The ReferItGame dataset."""

    def __init__(self, **kwargs):
        super().__init__()

        self.split_configs = ["refcoco", "refcoco+", "refcocog"]
        self.modality_names = ["image", "text"]

        self.data_name = Config().data.dataname
        self.base_coco = Config().data.base_coco_images_path
        self.data_source = "COCO2017"

        # Obtain which split to use:
        #  refclef, refcoco, refcoco+ and refcocog
        self.split_config = Config().data.split_config
        # Obtain which specific setting to use:
        #  unc, google
        self.split_name = Config().data.split_name
        if self.split_config not in self.split_configs:
            logging.info(
                "%s does not exist in the official configurations %s.",
                self.split_config, self.split_configs)

        _path = Config().params['data_path']
        self._data_path_process(data_path=_path, base_data_name=self.data_name)
        base_data_path = self.mm_data_info["data_path"]

        # raw coco images path
        coco_raw_imgs_path = self.base_coco
        if self._exists(coco_raw_imgs_path):
            logging.info(
                "Successfully connecting the source COCO2017 images data from the path %s",
                coco_raw_imgs_path)
        else:
            logging.info(
                "Fail to connect the source COCO2017 images data from the path %s",
                coco_raw_imgs_path)

        # download the public official code and the required config
        download_split_url = Config(
        ).data.download_splits_base_url + self.split_config + ".zip"
        for dd_url in [download_split_url]:
            self._download_arrange_data(download_url_address=dd_url,
                                        data_path=base_data_path)

        self._dataset_refer = referitgame_utils.REFER(
            data_root=base_data_path,
            image_dataroot=coco_raw_imgs_path,
            dataset=self.split_config,
            splitBy=self.split_name)  # default is unc or google

        self._splited_referids_holder = {}
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

        mode_elements_holder = {}
        mode_flatten_emelemts = []

        for refer_id in mode_refer_ids:

            ref = self._dataset_refer.loadRefs(refer_id)[0]
            image_id = ref['image_id']
            image_file_path = self._dataset_refer.loadImgspath(image_id)
            caption_phrases_cate = self._dataset_refer.Cats[ref['category_id']]
            caption_phrases_cate_id = ref['category_id']

            mode_elements_holder[refer_id] = {}
            mode_elements_holder[refer_id]["image_id"] = image_id
            mode_elements_holder[refer_id]["image_file_path"] = image_file_path

            mode_elements_holder[refer_id]["sentences"] = []
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

    def get_phase_dataset(self, phase, modality_sampler):
        """ Obtain the dataset for the specific phase """
        _, mode_flatten_emelemts = self.get_phase_data(phase)

        dataset = ReferItGameDataset(dataset_info=mode_flatten_emelemts,
                                     phase_info=self._dataset_refer,
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
