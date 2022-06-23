"""
The implementation of the SimCLR's [1] augmentation function.

The official code: https://github.com/google-research/simclr

The third-party code: https://github.com/PatrickHua/SimSiam


Reference:

[1]. https://arxiv.org/abs/2002.05709
"""

import torchvision.transforms as T


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
    rnd_gray = T.RandomGrayscale(p=0.2)
    color_distort = T.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class SimCLRTransform():
    """ This the contrastive data augmentation used by the SimCLR method. """

    def __init__(self, image_size, normalize):
        image_size = 224 if image_size is None else image_size

        transform_functions = [
            T.RandomResizedCrop(size=image_size),
            T.RandomHorizontalFlip(),  # with 0.5 probability
            get_color_distortion(s=0.5),
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
