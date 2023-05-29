"""Fill50K dataset, which contains a lot of circles."""
import os
import json
import cv2
import numpy as np

from torch.utils.data import Dataset


# pylint:disable=no-member
class Fill50KDataset(Dataset):
    """Fill 50k dataset"""

    def __init__(self, path):
        self.path = path
        self.data = []
        # pylint:disable=unspecified-encoding
        with open(os.path.join(path, "prompt.json"), "rt") as file:
            for line in file:
                self.data.append(json.loads(line))

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

        return {"jpg": target, "txt": prompt, "hint": source}, 0
