"""
The COCO dataset or other datasets for the YOLOv5 model.

For more information about COCO 128, which contains the first 128 images of the
COCO 2017 dataset, refer to https://www.kaggle.com/ultralytics/coco128.

For more information about the COCO 2017 dataset, refer to http://cocodataset.org.
"""

import logging
import os

import torch
from plato.config import Config
from yolov5.utils.datasets import LoadImagesAndLabels
from yolov5.utils.general import check_img_size

from plato.datasources import base


def collate_fn(batch):
    """Customized version for the MistNet server when cut_layer is used."""
    img, label = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0)


class YOLODataset(torch.utils.data.Dataset):
    """Prepares the YOLO dataset for use in YOLOv5."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label, paths, shapes = self.dataset[item]
        return image.float() / 255.0, label, paths, shapes


class DataSource(base.DataSource):
    """The YOLO dataset."""
    def __init__(self):
        super().__init__()
        _path = Config().data.data_path

        if not os.path.exists(_path):
            os.makedirs(_path)

            logging.info(
                "Downloading the YOLO dataset. This may take a while.")

            urls = Config().data.download_urls
            for url in urls:
                if not os.path.exists(_path + url.split('/')[-1]):
                    DataSource.download(url, _path)

        assert 'grid_size' in Config().params

        self.grid_size = Config().params['grid_size']
        self.image_size = check_img_size(Config().data.image_size,
                                         self.grid_size)

        self.train_set = None
        self.test_set = None

    def num_train_examples(self):
        return Config().data.num_train_examples

    def num_test_examples(self):
        return Config().data.num_test_examples

    def classes(self):
        """Obtains a list of class names in the dataset."""
        return Config().data.classes

    def get_train_set(self):
        single_class = (Config().data.num_classes == 1)

        if self.train_set is None:
            self.train_set = LoadImagesAndLabels(
                Config().data.train_path,
                self.image_size,
                Config().trainer.batch_size,
                augment=False,  # augment images
                hyp=None,  # augmentation hyperparameters
                rect=False,  # rectangular training
                cache_images=False,
                single_cls=single_class,
                stride=int(self.grid_size),
                pad=0.0,
                image_weights=False,
                prefix='')

        return self.train_set

    def get_test_set(self):
        single_class = (Config().data.num_classes == 1)

        if self.test_set is None:
            self.test_set = LoadImagesAndLabels(
                Config().data.test_path,
                self.image_size,
                Config().trainer.batch_size,
                augment=False,  # augment images
                hyp=None,  # augmentation hyperparameters
                rect=False,  # rectangular training
                cache_images=False,
                single_cls=single_class,
                stride=int(self.grid_size),
                pad=0.0,
                image_weights=False,
                prefix='')

        return self.test_set

    @staticmethod
    def get_train_loader(batch_size,
                         trainset,
                         sampler,
                         extract_features=False,
                         cut_layer=None):
        """The custom train loader for YOLOv5."""

        if extract_features:
            # MistNet client: feature extraction
            return torch.utils.data.DataLoader(dataset=YOLODataset(trainset),
                                               batch_size=batch_size,
                                               shuffle=False)
        elif cut_layer is not None:
            # MistNet server: training from the cut layer forwards using
            # the features extracted on the client
            return torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=collate_fn)
        else:
            return torch.utils.data.DataLoader(
                YOLODataset(trainset),
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=LoadImagesAndLabels.collate_fn)

    @staticmethod
    def get_test_loader(batch_size, testset):
        """The custom test loader for YOLOv5."""
        return torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=LoadImagesAndLabels.collate_fn)
