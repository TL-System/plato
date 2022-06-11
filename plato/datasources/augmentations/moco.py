"""
The implementation of the MoCo's [1] augmentation function.

The official code: https://github.com/facebookresearch/moco


Reference:

[1]. https://arxiv.org/abs/1911.05722
"""

import torchvision.transforms as T

# MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
# augmentation = [
#     transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize
# ]


class MoCoTransform():
    """ This the contrastive data augmentation used by the MoCo method. """

    def __init__(self, image_size, normalize):
        image_size = 224 if image_size is None else image_size

        transform_functions = [
            T.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
            T.RandomGrayscale(p=0.2),
            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]

        if normalize is not None:
            transform_functions.append(T.Normalize(*normalize))

        self.transform = T.Compose(transform_functions)

    def __call__(self, x):
        """ Perform the contrastive data augmentation. """
        x1 = self.transform(x)
        x2 = self.transform(x)

        return x1, x2
