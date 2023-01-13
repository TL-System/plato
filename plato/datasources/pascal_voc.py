"""
The PASCAL VOC dataset for image segmentation.
"""

from torchvision import datasets, transforms
from plato.config import Config

from plato.datasources import base


class DataSource(base.DataSource):
    """The PASCAL dataset."""

    def __init__(self, **kwargs):
        super().__init__()
        _path = Config().params["data_path"]
        self.mean = [0.45734706, 0.43338275, 0.40058118]
        self.std = [0.23965294, 0.23532275, 0.2398498]

        train_transform = (
            kwargs["train_transform"]
            if train_transform in kwargs
            else (
                transforms.Compose(
                    [
                        transforms.Resize((96, 96)),
                        transforms.ToTensor(),
                    ]
                )
            )
        )

        test_transform = train_transform

        self.trainset = datasets.VOCSegmentation(
            root=_path,
            year="2012",
            image_set="train",
            download=True,
            transform=train_transform,
            target_transform=train_transform,
        )
        self.testset = datasets.VOCSegmentation(
            root=_path,
            year="2012",
            image_set="val",
            download=True,
            transform=test_transform,
            target_transform=test_transform,
        )

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)
