"""
Basic dataset for applying other transfer.
"""
import abc

import cv2
from torch.utils.data import Dataset
import numpy as np

from ControlNet.annotator.uniformer import UniformerDetector
from ControlNet.annotator.openpose import OpenposeDetector
from ControlNet.annotator.midas import MidasDetector
from ControlNet.annotator.hed import HEDdetector
from ControlNet.annotator.mlsd import MLSDdetector
from ControlNet.annotator.canny import CannyDetector
from ControlNet.annotator.util import HWC3


class BasicDataset(Dataset):
    """The basic dataset"""

    def __init__(self, task=None, device="cpu") -> None:
        super().__init__()
        self.device = device
        if task is not None:
            assert task in [
                "canny",
                "depth",
                "hed",
                "hough",
                "normal",
                "pose",
                "scribble",
                "seg",
                "random",
            ]
        self.task = task

    @abc.abstractmethod
    def __getitem__(self, index):
        return

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
                if self.task == "depth" or self.task == "pose":
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
