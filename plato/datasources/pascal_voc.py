"""
The PASCAL VOC dataset for image segmentation.
"""

from torchvision import datasets, transforms
from plato.config import Config

from plato.datasources import base


class DataSource(base.DataSource):
    """The PASCAL dataset."""
    def __init__(self):
        super().__init__()
        _path = Config().data.data_path
        self.mean = [0.45734706, 0.43338275, 0.40058118]
        self.std = [0.23965294, 0.23532275, 0.2398498]
        _transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
        ])
        self.trainset = datasets.VOCSegmentation(root=_path,
                                                 year='2012',
                                                 image_set='train',
                                                 download=True,
                                                 transform=_transform,
                                                 target_transform=_transform)
        self.testset = datasets.VOCSegmentation(root=_path,
                                                year='2012',
                                                image_set='val',
                                                download=True,
                                                transform=_transform,
                                                target_transform=_transform)

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)
