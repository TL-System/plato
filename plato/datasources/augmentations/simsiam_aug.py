"""
The implementation of the simsiam's [1] augmentation function.

[1]. Chen & He, Exploring Simple Siamese Representation Learning, 2021.

This augmentaion is directly extracted from the 'augmentaions/' of
 https://github.com/PatrickHua/SimSiam.


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


"""

from plato.datasources.augmentations.ssl_transform_base import get_ssl_base_transform


class SimSiamTransform():
    """ This the contrastive data augmentation used by the Simsiam method. """

    def __init__(self, image_size, normalize):
        # by default simsiam use image size 224
        image_size = 224 if image_size is None else image_size
        p_blur = 0.5 if image_size > 32 else 0  # exclude cifar
        self.transform, transform_funcs = get_ssl_base_transform(
            image_size,
            normalize,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            color_jitter_prob=0.8,
            gray_scale_prob=0.2,
            horizontal_flip_prob=0.5,
            gaussian_prob=p_blur,
            solarization_prob=0.0,
            equalization_prob=0.0,
            min_scale=0.2,
            max_scale=1.0,
            crop_size=image_size)
        self.transform_funcs = transform_funcs

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
