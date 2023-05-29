"""Load CoCo dataset."""
import os
import json
import random

import cv2
import numpy as np

# pylint:disable=relative-beyond-top-level
from .dataset_basic import BasicDataset


# pylint:disable=no-member
class CoCoDataset(BasicDataset):
    """Coco dataset"""

    # pylint:disable=too-many-arguments
    def __init__(
        self,
        path,
        split,
        image_size=512,
        dataset_size=None,
        condition=None,
        device="cpu",
        index=0,
    ):
        super().__init__(condition, device)

        path_json = os.path.join(path, "annotations/captions_" + split + "2017.json")
        with open(path_json, "r", encoding="utf-8") as file:
            data = json.load(file)
        data = data["annotations"]
        self.files = []
        self.root_path_im = os.path.join(path, split + "2017")
        self.root_path_mask = os.path.join(path, split + "2017")
        for file in data:
            name = f"{file['image_id']:012d}.png"
            self.files.append({"name": name, "sentence": file["caption"]})
        if not (dataset_size is None or dataset_size >= self.__len__()):
            if not "val" in self.root_path_im:
                random.shuffle(self.files)
            self.files = self.files[
                index * dataset_size : min((index + 1) * dataset_size, self.__len__())
            ]
        self.image_size = image_size

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file["name"]
        image = cv2.imread(
            os.path.join(self.root_path_im, name.replace(".png", ".jpg"))
        )
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image.astype(np.float32) - 127.5) / 127.5

        if self.root_path_im == self.root_path_mask:
            mask = cv2.imread(
                os.path.join(self.root_path_mask, name.replace(".png", ".jpg"))
            )
        else:
            mask = cv2.imread(os.path.join(self.root_path_mask, name))  # [:,:,0]
        mask = cv2.resize(mask, (512, 512))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.process(mask)
        mask = mask.astype(np.float32) / 255.0

        sentence = file["sentence"]
        return {"jpg": image, "hint": mask, "txt": sentence}, 0

    def __len__(self):
        return len(self.files)
