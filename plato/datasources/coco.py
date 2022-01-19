"""
The MS COCO- dataset stands for Common Objects in Context, and is 
  designed to represent a vast array of objects that we 
  regularly encounter in everyday life.

We mainly utilize COCO-17 (25.20 GB):
 - COCO has 121,408 images in total.
 - has 883,331 object annotations
 - COCO defines 91 classes but the data only uses 80 classes.
 - Some images from the train and validation sets don’t have annotations.
 - The test set does not have annotations.
 - COCO 2014 and 2017 use the same images, but the splits are different.
 -> for image, detection, segmentation.

The data structure and setting follows:
 "https://cocodataset.org/#home".

Then, the download urls are obtained from:
 "https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9".

We utilize the official splits that contain:
 - train: 118,287 images
 - val: 5,000 images
 - test: 40,670 images

The file structure of this dataset is:
 - train2017: train images
 - test2017: test images
 - val2017: validation images
 - annotations_trainval2017: captions for train/val

The data structure under the 'data/' is:
├── COCO2017           # root dir of Flickr30K Entities dataset
│   ├── COCO2017Raw    # Raw images/annotations and the official splits
│   │   └── annotations
│   │   └── train2017
│   │   └── test2017
│   │   └── val2017
│   ├── train       # images for the train phase
│   └── test        # images for the test phase
│   └── val         # images for the validation phase

Note:
    Currently, we have not utilize the COCO dataset to train the model.
     Thus, we only implement the code of downloading and arrange the data,
     which is required when using the referitgame dataset.
"""

import os
import shutil

from plato.config import Config
from plato.datasources import multimodal_base


class DataSource(multimodal_base.MultiModalDataSource):
    """ The COCO dataset."""
    def __init__(self):
        super().__init__()

        self.data_name = Config().data.dataname
        self.data_source = Config().data.datasource

        self.modality_names = ["image", "text"]

        _path = Config().data.data_path
        self._data_path_process(data_path=_path, base_data_name=self.data_name)

        base_data_path = self.mm_data_info["base_data_dir_path"]
        raw_data_name = self.data_name + "Raw"
        raw_data_path = os.path.join(base_data_path, raw_data_name)
        if not self._exist_judgement(raw_data_path):
            os.makedirs(raw_data_path, exist_ok=True)

        download_train_url = Config().data.download_train_url
        download_test_url = Config().data.download_test_url
        download_val_url = Config().data.download_val_url
        download_annotation_url = Config().data.download_annotation_url

        splits_downalods = {
            "train": download_train_url,
            "test": download_test_url,
            "val": download_val_url
        }

        # Download raw data and extract to different splits
        for split_name in list(self.splits_info.keys()):
            split_path = self.splits_info[split_name]["path"]
            split_download_url = splits_downalods[split_name]
            split_file_name = self._download_arrange_data(
                download_url_address=split_download_url,
                put_data_dir=raw_data_path,
                extract_to_dir=split_path)
            # rename of the extracted file to "images"
            extracted_dir_path = os.path.join(split_path, split_file_name)
            rename_dir_path = os.path.join(split_path, "images")
            os.rename(src=extracted_dir_path, dst=rename_dir_path)

        # Download the annotation
        self._download_arrange_data(
            download_url_address=download_annotation_url,
            put_data_dir=raw_data_path)
        anno_dir_path = os.path.join(raw_data_path, "annotations")

        # Move the annotation to each split
        splits_caption_name = {
            "train": "captions_train2017.json",
            "val": "captions_val2017.json"
        }
        for split_name in list(splits_caption_name.keys()):
            split_caption_name = splits_caption_name[split_name]
            to_split_path = os.path.join(self.splits_info[split_name]["path"],
                                         "captions.json")
            shutil.copyfile(src=os.path.join(anno_dir_path,
                                             split_caption_name),
                            dst=to_split_path)
