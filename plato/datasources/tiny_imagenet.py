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
from plato.datasources import _image_folder, base


def _locate_layout(root: Path) -> tuple[Path, bool] | None:
    """Return the complete native or class-compatible prepared dataset layout."""
    for candidate in (root, root / "tiny-imagenet-200"):
        if not (candidate / "train").is_dir():
            continue
        annotation = candidate / "val/val_annotations.txt"
        if annotation.is_file() and (candidate / "val/images").is_dir():
            return candidate, False
        if not (candidate / "test").is_dir():
            continue
        training_classes = {
            folder.name for folder in (candidate / "train").iterdir() if folder.is_dir()
        }
        testing_classes = {
            folder.name for folder in (candidate / "test").iterdir() if folder.is_dir()
        }
        # Official test/images is unlabeled, even if validation was interrupted.
        if annotation.exists() or (
            "images" in testing_classes and "images" not in training_classes
        ):
            continue
        unknown = testing_classes - training_classes
        if unknown:
            raise ValueError(f"Unknown evaluation classes: {sorted(unknown)}")
        if testing_classes:
            return candidate, True
    return None


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
        layout = _locate_layout(root)
        if layout is None:
            incomplete_native = any(
                (candidate / "train").is_dir()
                and (
                    (candidate / "val").is_dir()
                    or (candidate / "test/images").is_dir()
                )
                for candidate in (root, root / "tiny-imagenet-200")
            )
            completed_default_download = (
                root / "tiny-imagenet-200.zip.complete"
            ).is_file()
            if (
                incomplete_native
                and not hasattr(Config().data, "download_url")
                and not completed_default_download
            ):
                raise ValueError(
                    "Incomplete native Tiny ImageNet validation data: need "
                    "val/val_annotations.txt and val/images. Official test/images "
                    "is unlabeled; provide a download_url to recover the dataset."
                )
            logging.info(
                "Downloading the Tiny ImageNet 200 dataset. This may take a while."
            )
            url = (
                Config().data.download_url
                if hasattr(Config().data, "download_url")
                else "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
            )
            DataSource.download(
                url, _path, ready=lambda: _locate_layout(root) is not None
            )
            layout = _locate_layout(root)
            if layout is None:
                raise ValueError("Incomplete Tiny ImageNet dataset after download.")

        root, prepared = layout

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
        test_transform = kwargs.get(
            "test_transform",
            transforms.Compose(
                [
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        )
        self.trainset = datasets.ImageFolder(
            root=str(root / "train"), transform=train_transform
        )
        if prepared:
            self.testset = _image_folder.evaluation_folder(
                root / "test", self.trainset, transform=test_transform
            )
        else:
            # The official test images have no public labels; evaluate on the
            # annotated validation split rather than treating "images" as a class.
            self.testset = ValidationDataset(root / "val", self.trainset.classes,
                                             transform=test_transform)
