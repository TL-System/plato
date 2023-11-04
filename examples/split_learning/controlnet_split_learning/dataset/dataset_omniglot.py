"""Omniglot dataset, which contains a lot of circles."""
import os
from typing import Optional, Callable
import copy

import cv2
import numpy as np


from torchvision.datasets import Omniglot

# pylint:disable=import-error
from ControlNet.annotator.uniformer import UniformerDetector
from ControlNet.annotator.openpose import OpenposeDetector
from ControlNet.annotator.midas import MidasDetector
from ControlNet.annotator.hed import HEDdetector
from ControlNet.annotator.mlsd import MLSDdetector
from ControlNet.annotator.canny import CannyDetector
from ControlNet.annotator.util import HWC3


# pylint:disable=no-member
class OmniglotDataset(
    Omniglot,
):
    """Fill 50k dataset"""

    # pylint:disable=too-many-arguments
    def __init__(
        self,
        root: str,
        background: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        condition=None,
    ) -> None:
        super().__init__(
            root,
            background,
            transform,
            target_transform,
            True,
        )
        self.task = condition

    def __getitem__(self, index):
        image_name, character_class = self._flat_character_images[index]
        image_path = os.path.join(
            self.target_folder, self._characters[character_class], image_name
        )

        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = copy.deepcopy(image)
        image = (image.astype(np.float32) - 127.5) / 127.5

        mask = self.process(mask)
        mask = mask.astype(np.float32) / 255.0

        return {"jpg": image, "txt": "Good image", "hint": mask}, 0

    def process(self, condition):
        """To generate the condition according to the task."""
        if self.task is None:
            return condition
        if self.task == "random":
            mean = np.mean(condition, axis=(0, 1))
            std = np.std(condition, axis=(0, 1))
            noise = np.random.normal(mean, std, condition.shape)
            noise = np.clip(noise * 127.5 + 127.5, 0, 255)
            return noise
        if self.task == "scribble":
            detected_map = np.zeros_like(condition, dtype=np.uint8)
            detected_map[np.min(condition, axis=2) < 127] = 255
            return detected_map
        if self.task in ["seg", "pose", "normal", "hough", "hed", "canny", "depth"]:
            operators = {
                "seg": UniformerDetector,
                "depth": MidasDetector,
                "pose": OpenposeDetector,
                "normal": MidasDetector,
                "hough": MLSDdetector,
                "hed": HEDdetector,
                "canny": CannyDetector,
            }
            height, width, _ = condition.shape
            if self.task == "canny":
                detected_map = operators[self.task]()(condition, 100, 200)
                detected_map = HWC3(detected_map)
            else:
                if self.task in ["depth", "pose"]:
                    detected_map, _ = operators[self.task]()(condition)
                elif self.task == "hough":
                    detected_map = operators[self.task]()(condition, 0.1, 0.1)
                elif self.task == "normal":
                    _, detected_map = operators[self.task]()(condition, 0.4)
                    detected_map = detected_map[:, :, ::-1]
                else:
                    detected_map = operators[self.task]()(condition)
                detected_map = HWC3(detected_map)
                detected_map = cv2.resize(
                    detected_map, (height, width), interpolation=cv2.INTER_NEAREST
                )
            return detected_map
        return condition
