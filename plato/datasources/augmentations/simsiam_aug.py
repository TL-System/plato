"""
The implementation of the simsiam's [1] augmentation function.

[1]. Chen & He, Exploring Simple Siamese Representation Learning, 2021.

This augmentaion is directly extracted from the 'augmentaions/' of 
 https://github.com/PatrickHua/SimSiam.

"""

import torchvision.transforms as T


class SimSiamTransform():

    def __init__(self, image_size, normalize):
        image_size = 224 if image_size is None else image_size  # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0  # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        transform_functions = [
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([
                T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1,
                               sigma=(0.1, 2.0))
            ],
                          p=p_blur),
            T.ToTensor()
        ]

        if normalize is not None:
            transform_functions.append(T.Normalize(*normalize))

        self.transform = T.Compose(transform_functions)

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
