"""Quick script to evaluate an MLX client checkpoint on MNIST.

Usage:
    uv run scripts/validate_mlx_client.py \
        --checkpoint runtime/models/pretrained/lenet5_2_49258.safetensors
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from plato.models.mlx.lenet5 import Model
from plato.serialization.safetensor import deserialize_tree
from plato.trainers import mlx as mlx_trainer
from plato.utils import fonts


def load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        blob = f.read()
    return deserialize_tree(blob)


def mnist_loader(batch_size: int = 128):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    testset = datasets.MNIST(
        root="./runtime/data", train=False, download=True, transform=transform
    )
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return loader


def ensure_nhwc(array: mx.array) -> mx.array:
    if array.ndim == 4:
        channels_first = array.shape[1] <= 4 and array.shape[-1] > 4
        if channels_first:
            return mx.transpose(array, (0, 2, 3, 1))
    return array


def evaluate(model: Model, loader) -> float:
    total = 0
    correct = 0
    for images, labels in loader:
        images = images.numpy()
        labels = labels.numpy()

        images = mx.array(images)
        labels = mx.array(labels)

        images = ensure_nhwc(images)

        logits = model(images)
        preds = mx.argmax(logits, axis=-1)

        correct += int(mx.sum(preds == labels).item())
        total += labels.shape[0]

    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    print(fonts.colourize(f"Loading checkpoint: {args.checkpoint}"))
    state = load_checkpoint(args.checkpoint)

    model = Model(num_classes=10)
    restored = mlx_trainer._tree_map(mlx_trainer._to_mx_array, state)
    model.update(restored)
    if hasattr(mx, "eval"):
        leaves = [
            leaf
            for leaf in mlx_trainer._tree_leaves(model.parameters())
            if isinstance(leaf, mx.array)
        ]
        if leaves:
            mx.eval(*leaves)

    loader = mnist_loader()
    accuracy = evaluate(model, loader)
    print(fonts.colourize(f"Accuracy: {accuracy * 100:.2f}%"))


if __name__ == "__main__":
    main()
