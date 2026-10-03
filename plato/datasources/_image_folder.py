"""Prepared evaluation folders sharing the training dataset's label space."""

from collections.abc import Callable
from pathlib import Path

from torchvision import datasets


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
