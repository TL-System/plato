"""
The implementation of data augmentation for the eval/test phase
of contrastive learning.

In general, the evaluation phase is implemented by the
linear evaluation utilizing the general transform.

"""

from torchvision import transforms as T
from PIL import Image


class TestTransform():
    """ The transform for test and evaluation. """

    def __init__(self, image_size, train, normalize):
        transform_functions = []
        if train:
            transform_functions = [
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor()
            ]
        else:
            transform_functions = [
                #T.Resize(image_size),  # 224 -> 256
                T.ToTensor()
            ]

        if normalize is not None:

            transform_functions.append(T.Normalize(*normalize))

        self.transform = T.Compose(transform_functions)
        self.transform_funcs = transform_functions

    def __call__(self, x):

        return self.transform(x)