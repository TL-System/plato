"""
The CelebA dataset from the torchvision package.
"""
import logging
import os
from typing import Callable, List, Optional, Union

import torch
from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import base


class CelebA(datasets.CelebA):
    """
    A wrapper class of torchvision's CelebA dataset class
    to add <targets> and <classes> attributes as celebrity
    identity, which is used for non-IID samplers.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root, split, target_type, transform, target_transform, download
        )
        self.targets = self.identity.flatten().tolist()
        self.classes = [f"Celebrity #{i}" for i in range(10177 + 1)]


class DataSource(base.DataSource):
    """The CelebA dataset."""

    def __init__(self, **kwargs):
        super().__init__()
        _path = Config().params["data_path"]

        if not os.path.exists(os.path.join(_path, "celeba")):
            celeba_url = "http://iqua.ece.toronto.edu/baochun/celeba.tar.gz"
            DataSource.download(celeba_url, _path)
        else:
            logging.info(
                "CelebA data already decompressed under %s",
                os.path.join(_path, "celeba"),
            )

        target_types = []
        if hasattr(Config().data, "celeba_targets"):
            targets = Config().data.celeba_targets
            if hasattr(targets, "attr") and targets.attr:
                target_types.append("attr")
            if hasattr(targets, "identity") and targets.identity:
                target_types.append("identity")
        else:
            target_types = ["attr", "identity"]

        image_size = 64
        if hasattr(Config().data, "celeba_img_size"):
            image_size = Config().data.celeba_img_size

        train_transform = (
            kwargs["train_transform"]
            if "train_transform" in kwargs
            else (
                transforms.Compose(
                    [
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )
            )
        )

        test_transform = train_transform

        target_transform = (
            kwargs["target_transform"]
            if "target_transform" in kwargs
            else (DataSource._target_transform if target_types else None)
        )

        self.trainset = CelebA(
            root=_path,
            split="train",
            target_type=target_types,
            download=False,
            transform=train_transform,
            target_transform=target_transform,
        )
        self.testset = CelebA(
            root=_path,
            split="test",
            target_type=target_types,
            download=False,
            transform=test_transform,
            target_transform=target_transform,
        )

    @staticmethod
    def _target_transform(label):
        """
        Output labels are in a tuple of tensors if specified more
        than one target types, so we need to convert the tuple to
        tensors. Here, we just merge two tensors by adding identity
        as the 41st attribute
        """
        if isinstance(label, tuple):
            if len(label) == 1:
                return label[0]
            elif len(label) == 2:
                attr, identity = label
                return torch.cat(
                    (
                        attr.reshape(
                            [
                                -1,
                            ]
                        ),
                        identity.reshape(
                            [
                                -1,
                            ]
                        ),
                    )
                )
        else:
            return label

    @staticmethod
    def input_shape():
        image_size = 64
        if hasattr(Config().data, "celeba_img_size"):
            image_size = Config().data.celeba_img_size
        return [162770, 3, image_size, image_size]

    def num_train_examples(self):
        return 162770

    def num_test_examples(self):
        return 19962
