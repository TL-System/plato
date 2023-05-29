"""The dataset used for experiments in ControlNet"""
import os

from plato.config import Config
from plato.datasources import base

from dataset.dataset_celeba import CelebADataset
from dataset.dataset_coco import CoCoDataset
from dataset.dataset_fill50k import Fill50KDataset
from dataset.dataset_omniglot import OmniglotDataset


class DataSource(base.DataSource):
    """The datasource class specifiedly used for ControlNet privacy study."""

    def __init__(self):
        super().__init__()
        _path = Config().params["data_path"]
        _condition = Config().data.condition
        _val_dataset = Config().data.val_dataset

        self.trainset = CoCoDataset(
            os.path.join(_path, "coco"),
            "train",
            condition=_condition,
            dataset_size=50000,
        )
        if _val_dataset == "celeba":
            testset = CelebADataset(os.path.join(_path, "celeba"), "valid", _condition)
        elif _val_dataset == "coco":
            testset = CoCoDataset(
                os.path.join(_path, "coco"),
                split="val",
                condition=_condition,
                dataset_size=1000,
            )
        elif _val_dataset == "fill50k":
            testset = Fill50KDataset(os.path.join(_path, "fill50k"))
        elif _val_dataset == "omniglot":
            testset = OmniglotDataset(
                root=os.path.join(_path, "omniglot"), condition=_condition
            )
        else:
            raise ValueError("Dataset does not exist")
        self.testset = testset

    @staticmethod
    def input_shape():
        return 512

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)
