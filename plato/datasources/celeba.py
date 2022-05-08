"""
The CelebA dataset from the torchvision package.
"""

import torch
from typing import Callable, List, Optional, Union
from torchvision import datasets, transforms
from plato.config import Config
from plato.datasources import base


class CelebA(datasets.CelebA):
    """
    A wrapper class of torchvision's CelebA dataset class
    to add <targets> and <classes> attributes as celebrity
    identity, which is used for non-IID samplers.
    """

    def __init__(self,
                 root: str,
                 split: str = "train",
                 target_type: Union[List[str], str] = "attr",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root, split, target_type, transform, target_transform,
                         download)
        self.targets = self.identity.flatten().tolist()
        self.classes = [f'Celebrity #{i}' for i in range(10177 + 1)]


class DataSource(base.DataSource):
    """The CelebA dataset."""

    def __init__(self):
        super().__init__()
        _path = Config().data.data_path

        image_size = 64
        _transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.trainset = CelebA(root=_path,
                               split='train',
                               target_type=['attr', 'identity'],
                               download=True,
                               transform=_transform,
                               target_transform=DataSource._target_transform)
        self.testset = CelebA(root=_path,
                              split='test',
                              target_type=['attr', 'identity'],
                              download=True,
                              transform=_transform,
                              target_transform=DataSource._target_transform)

    @staticmethod
    def _target_transform(label):
        """
        Output labels are in a tuple of tensors if specified more
        than one target types, so we need to convert the tuple to
        tensors. Here, we just merge two tensors by adding identity
        as the 41st attribute
        """
        attr, identity = label
        return torch.cat((attr.reshape([
            -1,
        ]), identity.reshape([
            -1,
        ])))

    @staticmethod
    def input_shape():
        return [162770, 2, 64, 64]

    def num_train_examples(self):
        return 162770

    def num_test_examples(self):
        return 19962
