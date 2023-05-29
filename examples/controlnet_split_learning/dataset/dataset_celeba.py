"""Load CoCo dataset."""
import os
import csv
from collections import namedtuple
from typing import Optional
import copy

import cv2
from torchvision.datasets.utils import verify_str_arg
import torch
import numpy as np

# pylint:disable=relative-beyond-top-level
from .dataset_basic import BasicDataset


CSV = namedtuple("CSV", ["header", "index", "data"])


# pylint:disable=no-member
class CelebADataset(BasicDataset):
    """Coco dataset"""

    def __init__(
        self,
        path,
        split,
        task,
    ):
        super().__init__(task=task)
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        self.root = path
        split_ = split_map[
            verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))
        ]
        splits = self._load_csv("list_eval_partition.txt")

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):
            self.filename = splits.index
        else:
            self.filename = [
                splits.index[i] for i in torch.squeeze(torch.nonzero(mask))
            ]
        attr = self._load_csv("list_attr_celeba.txt", header=1)
        self.attr = attr.data[mask]
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")

    def __getitem__(self, idx):
        image = cv2.imread(
            os.path.join(self.root, "img_align_celeba", self.filename[idx])
        )
        mask = copy.deepcopy(image)
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image.astype(np.float32) - 127.5) / 127.5

        mask = cv2.resize(mask, (512, 512))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.process(mask)
        mask = mask.astype(np.float32) / 255.0

        sentence = "Good image"
        return {"jpg": image, "hint": mask, "txt": sentence}, 0

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        # pylint:disable=unspecified-encoding
        with open(os.path.join(self.root, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]
        return CSV(headers, indices, torch.tensor(data_int))

    def __len__(self) -> int:
        return len(self.attr)
