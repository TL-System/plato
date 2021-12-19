"""A trainer for NNRT."""

import copy
from typing import Tuple
import numpy as np
from plato.trainers import base
from plato.utils import unary_encoding


class Trainer(base.Trainer):
    """A trainer for NNRT."""
    def __init__(self, model=None):
        super().__init__()

        # TODO: if the model is none, we get model from registry
        self.model = model

    def save_model(self, filename=None):
        pass

    def load_model(self, filename=None):
        pass

    def train(self, trainset, sampler, cut_layer=None) -> Tuple[bool, float]:
        pass

    def test(self, testset) -> float:
        pass

    async def server_test(self, testset):
        pass

    def randomize(self, bit_array: np.ndarray, targets: np.ndarray, epsilon):
        """
        The object detection unary encoding method.
        """
        assert isinstance(bit_array, np.ndarray)
        img = unary_encoding.symmetric_unary_encoding(bit_array, 1)
        label = unary_encoding.symmetric_unary_encoding(bit_array, epsilon)
        targets_new = copy.deepcopy(targets)
        for i in range(targets_new.shape[1]):
            box = self.convert(bit_array.shape[2:], targets_new[0][i][2:])
            img[:, :, box[0]:box[2],
                box[1]:box[3]] = label[:, :, box[0]:box[2], box[1]:box[3]]
        return img

    def convert(self, size, box):
        """The convert for YOLOv5.
              Arguments:
                  size: Input feature size(w,h)
                  box:(xmin,xmax,ymin,ymax).
              """
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        x1 = max(x - 0.5 * w - 3, 0)
        x2 = min(x + 0.5 * w + 3, size[0])
        y1 = max(y - 0.5 * h - 3, 0)
        y2 = min(y + 0.5 * h + 3, size[1])

        x1 = round(x1 * size[0])
        x2 = round(x2 * size[0])
        y1 = round(y1 * size[1])
        y2 = round(y2 * size[1])

        return (int(x1), int(y1), int(x2), int(y2))
