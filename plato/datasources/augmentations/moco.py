"""
The implementation of the MoCo's [1] augmentation function.

The official code: https://github.com/facebookresearch/moco


Reference:

[1]. https://arxiv.org/abs/1911.05722
"""

# MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
# augmentation = [
#     transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize
# ]

from plato.datasources.augmentations.visual_augmentations import get_visual_transform


class MoCoTransform():
    """ This the contrastive data augmentation used by the MoCo method. """

    def __init__(self, image_size, normalize):
        image_size = 224 if image_size is None else image_size
        self.transform, transform_funcs = get_visual_transform(
            image_size,
            normalize,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            color_jitter_prob=0.8,
            gray_scale_prob=0.2,
            horizontal_flip_prob=0.5,
            gaussian_prob=0.0,
            solarization_prob=0.0,
            equalization_prob=0.0,
            min_scale=0.08,
            max_scale=1.0,
            crop_size=image_size)
        self.transform_funcs = transform_funcs

    def __call__(self, x):
        """ Perform the contrastive data augmentation. """
        x1 = self.transform(x)
        x2 = self.transform(x)

        return x1, x2
