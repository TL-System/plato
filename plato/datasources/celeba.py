"""
The CelebA dataset from the torchvision package.
"""

from typing import Callable, List, Optional, Union
from torchvision import datasets, transforms
from plato.config import Config
from plato.datasources import base


class CelebA(datasets.CelebA):

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

        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.trainset = CelebA(
            root=_path,
            split='train',
            target_type=['attr', 'identity', 'bbox', 'landmarks'],
            download=True,
            transform=_transform)
        self.testset = CelebA(
            root=_path,
            split='test',
            target_type=['attr', 'identity', 'bbox', 'landmarks'],
            download=True,
            transform=_transform)

    def num_train_examples(self):
        return 162770

    def num_test_examples(self):
        return 19962
