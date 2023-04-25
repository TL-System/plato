"""
The STL-10 dataset from the torchvision package.
The details of this data can be found on the websites:
https://cs.stanford.edu/~acoates/stl10/ 
and
https://www.kaggle.com/datasets/jessicali9530/stl10
.
"""

from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

from plato.config import Config
from plato.datasources import base


class STL10Dataset(Dataset):
    """Prepares the STL10 dataset for the subsequence usage.
    The class annotation of the STL10 dataset is denoted as
    labels instead of targets used by subsequence learning
    of the Plato.
    """

    def __init__(self, dataset):
        self.dataset = dataset

        # obtain the raw data for subsequence
        # usage, such as the self-supervised learning
        self.data = self.dataset.data
        self.targets = self.dataset.labels
        self.target_transform = self.dataset.target_transform
        self.classes = self.dataset.classes

    def __getitem__(self, index):

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class DataSource(base.DataSource):
    """The STL-10 dataset."""

    def __init__(self, **kwargs):
        super().__init__()
        _path = Config().params["data_path"]

        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        train_transform = (
            kwargs["train_transform"]
            if "train_transform" in kwargs
            else (
                transforms.Compose(
                    [
                        transforms.RandomCrop(96, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            )
        )

        test_transform = (
            kwargs["test_transform"]
            if "test_transform" in kwargs
            else (
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            )
        )

        stl10_trainset = datasets.STL10(
            root=_path, split="train", download=True, transform=train_transform
        )
        stl10_unlabeled_set = datasets.STL10(
            root=_path, split="unlabeled", download=True, transform=train_transform
        )

        stl10_testset = datasets.STL10(
            root=_path, split="test", download=True, transform=test_transform
        )

        self.trainset = STL10Dataset(stl10_trainset)
        self.unlabeledset = STL10Dataset(stl10_unlabeled_set)
        self.testset = STL10Dataset(stl10_testset)

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)

    def get_unlabeled_set(self):
        """Obtains the unlabeled dataset."""
        return self.unlabeledset
