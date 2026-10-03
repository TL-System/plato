"""
The Tiny ImageNet 200 Classification dataset.

Tiny ImageNet contains 100000 images of 200 classes (500 for each class)
downsized to 64×64 colored images.
Each class has 500 training images, 50 validation images and 50 test images.
"""

import logging
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from plato.config import Config
from plato.datasources import base


class ValidationDataset(Dataset):
    """Read the canonical labeled validation split using training class IDs."""

    def __init__(self, root: Path, classes: list[str], transform=None):
        self.classes = list(classes)
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        self.transform = transform
        self.samples = []
        with (root / "val_annotations.txt").open(encoding="utf-8") as annotations:
            for line in annotations:
                fields = line.split()
                if len(fields) < 2 or fields[1] not in self.class_to_idx:
                    raise ValueError(
                        f"Invalid Tiny ImageNet validation annotation: {line!r}"
                    )
                filename, label = fields[:2]
                if Path(filename).name != filename:
                    raise ValueError(
                        f"Invalid Tiny ImageNet image filename: {filename!r}"
                    )
                self.samples.append((str(root / "images" / filename),
                                     self.class_to_idx[label]))
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        filename, label = self.samples[index]
        image = default_loader(filename)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class DataSource(base.DataSource):
    """The Tiny ImageNet 200 dataset."""

    def __init__(self, **kwargs):
        super().__init__()
        _path = Config().params["data_path"]

        root = Path(_path)
        native = (root / "val/val_annotations.txt").is_file()
        prepared = (
            (root / "train").is_dir() and (root / "test").is_dir() and not native
        )
        canonical = root if native else root / "tiny-imagenet-200"
        if not prepared and not (
            (canonical / "train").is_dir()
            and (canonical / "val/val_annotations.txt").is_file()
        ):
            logging.info(
                "Downloading the Tiny ImageNet 200 dataset. This may take a while."
            )
            url = (
                Config().data.download_url
                if hasattr(Config().data, "download_url")
                else "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
            )
            DataSource.download(url, _path)
            native = (root / "val/val_annotations.txt").is_file()
            prepared = (
                (root / "train").is_dir() and (root / "test").is_dir() and not native
            )
            canonical = root if native else root / "tiny-imagenet-200"

        train_transform = (
            kwargs["train_transform"]
            if "train_transform" in kwargs
            else (
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                )
            )
        )
        test_transform = train_transform
        if not prepared:
            root = canonical
        self.trainset = datasets.ImageFolder(
            root=str(root / "train"), transform=train_transform
        )
        if prepared:
            self.testset = datasets.ImageFolder(
                root=str(root / "test"), transform=test_transform
            )
        else:
            # The official test images have no public labels; evaluate on the
            # annotated validation split rather than treating "images" as a class.
            self.testset = ValidationDataset(root / "val", self.trainset.classes,
                                             transform=test_transform)
