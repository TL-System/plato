""" The implementation of data augmentation for the eval phase 
    of contrastive learning.
    
    In general, the evaluation phase is implemented by the 
        linear evaluation. This is a typical transform.

"""

from torchvision import transforms as T
from PIL import Image


class TestTransform():

    def __init__(self, image_size, train, normalize):
        transform_functions = []
        if train == True:
            transform_functions = [
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor()
            ]
        else:
            transform_functions = [
                T.Resize(image_size),  # 224 -> 256 
                T.CenterCrop(image_size),
                T.ToTensor()
            ]

        if normalize is not None:
            transform_functions.append(T.Normalize(*normalize))

        self.transform = T.Compose(transform_functions)

    def __call__(self, x):

        return self.transform(x)


class ByolTestTransform():

    def __init__(self, image_size, train, normalize):
        #self.denormalize = Denormalize(*imagenet_norm)
        transform_functions = []
        if train == True:
            transform_functions = [
                T.RandomResizedCrop(image_size,
                                    scale=(0.08, 1.0),
                                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                                    interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ToTensor()
            ]

        else:
            transform_functions = [
                T.Resize(int(image_size * (8 / 7)),
                         interpolation=Image.BICUBIC),  # 224 -> 256 
                T.CenterCrop(image_size),
                T.ToTensor()
            ]

        if normalize is not None:
            transform_functions.append(T.Normalize(*normalize))

        self.transform = T.Compose(transform_functions)

    def __call__(self, x):
        return self.transform(x)
