"""Prepared evaluation folders sharing the training dataset's label space."""

from collections.abc import Callable
from pathlib import Path

from torchvision import datasets
from torchvision.datasets.folder import find_classes, is_image_file, make_dataset


def folder_ready(root: str | Path) -> bool:
    """Check that every present class has an existing supported image file."""
    try:
        return bool(
            make_dataset(
                str(root),
                is_valid_file=lambda path: Path(path).is_file() and is_image_file(path),
            )
        )
    except FileNotFoundError:
        return False


def prepared_ready(root: str | Path) -> bool:
    """Check usable train/test folders without accepting unknown test classes."""
    root = Path(root)
    try:
        training_classes, _ = find_classes(root / "train")
        testing_classes, _ = find_classes(root / "test")
    except FileNotFoundError:
        return False
    unknown = set(testing_classes) - set(training_classes)
    if unknown:
        raise ValueError(f"Unknown evaluation classes: {sorted(unknown)}")
    return folder_ready(root / "train") and folder_ready(root / "test")


def evaluation_folder(
    root: str | Path, training: datasets.ImageFolder, transform: Callable | None = None
) -> datasets.ImageFolder:
    """Load labeled evaluation subsets and align all metadata to training IDs."""
    dataset = datasets.ImageFolder(root=str(root), transform=transform)
    unknown = set(dataset.classes) - set(training.classes)
    if unknown:
        raise ValueError(f"Unknown evaluation classes: {sorted(unknown)}")
    samples = [
        (filename, training.class_to_idx[dataset.classes[label]])
        for filename, label in dataset.samples
    ]
    dataset.classes = list(training.classes)
    dataset.class_to_idx = dict(training.class_to_idx)
    dataset.samples = dataset.imgs = samples
    dataset.targets = [label for _, label in samples]
    return dataset
