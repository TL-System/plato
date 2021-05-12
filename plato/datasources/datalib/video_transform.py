""" 
The transformers for the video classification task
"""

import torch
import torch.nn as nn
from torchvision.transforms import transforms


class ConvertBHWCtoBCHW(nn.Module):
    """Convert tensor from (B, H, W, C) to (B, C, H, W)
    """
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)


class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)
    """
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class VideoClassificationTrainTransformer:
    def __init__(self,
                 resize_size,
                 crop_size,
                 mean=(0.43216, 0.394666, 0.37645),
                 std=(0.22803, 0.22145, 0.216989),
                 hflip_prob=0.5):
        trans = [
            ConvertBHWCtoBCHW(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resize_size),
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomCrop(crop_size),
            ConvertBCHWtoCBHW()
        ])
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationEvalTransformer:
    def __init__(self,
                 resize_size,
                 crop_size,
                 mean=(0.43216, 0.394666, 0.37645),
                 std=(0.22803, 0.22145, 0.216989)):
        self.transforms = transforms.Compose([
            ConvertBHWCtoBCHW(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(resize_size),
            transforms.Normalize(mean=mean, std=std),
            transforms.CenterCrop(crop_size),
            ConvertBCHWtoCBHW()
        ])

    def __call__(self, x):
        return self.transforms(x)
