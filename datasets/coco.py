"""
The COCO dataset.

For more information about COCO 128, which contains the first 128 images of the
COCO 2017 dataset, refer to https://www.kaggle.com/ultralytics/coco128.

For more information about the COCO 2017 dataset, refer to http://cocodataset.org.
"""

import os
import logging

from torchvision import datasets

from config import Config
from datasets import base
from utils.yolov5.datasets import create_dataloader

class Dataset(base.Dataset):
    """The COCO dataset."""
    def __init__(self, path):
        super().__init__(path)

        if not os.path.exists(path):
            os.makedirs(path)

        logging.info("Downloading the COCO dataset. This may take a while.")

        urls = Config().data.download_urls
        for url in urls:
            if not os.path.exists(path + url.split('/')[-1]):
                Dataset.download(url, path)

            train_path = Config().data.train_path
            imgsz = None
            batch_size = Config().trainer.batch_size
            gs = None
            opt = None
            dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt)

    @staticmethod
    def num_train_examples():
        return Config().data.num_train_examples

    @staticmethod
    def num_test_examples():
        return Config().data.num_test_examples

    @staticmethod
    def num_classes():
        return Config().data.num_classes

    def get_train_set(self):
        return datasets.ImageFolder(root=self.coco_path + '/train',
                                    transform=self._transform)

    def get_test_set(self):
        return datasets.ImageFolder(root=self.coco_path + '/test',
                                    transform=self._transform)
