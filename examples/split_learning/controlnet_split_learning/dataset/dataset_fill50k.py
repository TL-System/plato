"""Fill50K dataset, which contains a lot of circles."""
import os
import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from plato.datasources.base import DataSource

# pylint:disable=relative-beyond-top-level
from .dataset_basic import DiffusionInputs


# pylint:disable=no-member
class Fill50KDataset(Dataset):
    """Fill 50k dataset"""

    def __init__(self, path, split="train"):
        # download the dataset if it does not exist.
        if not os.path.exists(path):
            url = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/training/fill50k.zip"
            DataSource.download(url, path)
        path = os.path.join(path, "fill50k")
        self.path = path
        self.data = []
        # pylint:disable=unspecified-encoding
        with open(os.path.join(path, "prompt.json"), "rt") as file:
            for line in file:
                self.data.append(json.loads(line))
        dataset_size = len(self.data)
        if split == "train":
            self.data = self.data[: int(0.8 * dataset_size)]
        else:
            self.data = self.data[int(0.8 * dataset_size) :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item["prompt"]

        source = cv2.imread(os.path.join(self.path, source_filename))
        target = cv2.imread(os.path.join(self.path, target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        inputs = DiffusionInputs()
        inputs["jpg"] = target
        inputs["hint"] = source
        inputs["txt"] = prompt
        return inputs, 0
