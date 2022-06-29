"""
The implementation of the FedEMA's augmentation function.

This augmentation is the same as the one of BYOL.

 [1]. Jean-Bastien Grill,, Bootstrap your own latent: A new approach
 to self-supervised Learning, 2021.
 Source code: https://github.com/lucidrains/byol-pytorch

This augmentaion is directly extracted from the 'augmentaions/' of
 https://github.com/PatrickHua/SimSiam.

"""

from plato.datasources.augmentations.ssl_transform_base import get_ssl_base_transform


class ContrastiveAdapTransform():
    """ This the contrastive data augmentation used by the FedEMA method. """

    def __init__(self, image_size, normalize):
        self.transform1, transform_funcs1 = get_ssl_base_transform(
            image_size,
            normalize,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            color_jitter_prob=0.8,
            gray_scale_prob=0.2,
            horizontal_flip_prob=0.5,
            gaussian_prob=1.0,
            solarization_prob=0.0,
            equalization_prob=0.0,
            min_scale=0.08,
            max_scale=1.0,
            crop_size=image_size)

        self.transform2, transform_funcs2 = get_ssl_base_transform(
            image_size,
            normalize,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            color_jitter_prob=0.8,
            gray_scale_prob=0.2,
            horizontal_flip_prob=0.5,
            gaussian_prob=0.1,
            solarization_prob=0.2,
            equalization_prob=0.0,
            min_scale=0.08,
            max_scale=1.0,
            crop_size=image_size)

        self.transform_funcs = [transform_funcs1, transform_funcs2]

    def __call__(self, x):
        """ Perform data augmentation. """

        x1 = self.transform1(x)
        x2 = self.transform2(x)
        return x1, x2


class Solarization():
    """ Behave as the Image Filter """

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, image):
        return ImageOps.solarize(image, self.threshold)
