"""
The implementation of the BYLO's [1] augmentation function.

 [1]. Jean-Bastien Grill,, Bootstrap your own latent: A new approach
 to self-supervised Learning, 2021.
 Source code: https://github.com/lucidrains/byol-pytorch

This augmentaion is directly extracted from the 'augmentaions/' of
 https://github.com/PatrickHua/SimSiam.


One template is:

        transform_functions1 = [
            T.RandomResizedCrop(image_size,
                                scale=(0.08, 1.0),
                                ratio=(3.0 / 4.0, 4.0 / 3.0),
                                interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            # simclr paper gives the kernel size. Kernel size has to be
            # odd positive number with torchvision
            T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1,
                           sigma=(0.1, 2.0)),
            T.ToTensor()
        ]

        transform_functions2 = [
            T.RandomResizedCrop(image_size,
                                scale=(0.08, 1.0),
                                ratio=(3.0 / 4.0, 4.0 / 3.0),
                                interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur(
            # kernel_size=int(0.1 * image_size))], p=0.1),
            T.RandomApply([
                T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1,
                               sigma=(0.1, 2.0))
            ],
                          p=0.1),
            T.RandomApply([Solarization()], p=0.2),
            T.ToTensor()
        ]
        if normalize is not None:
            transform_functions1.append(T.Normalize(*normalize))
            transform_functions2.append(T.Normalize(*normalize))


"""

from ssl_transform_base import get_ssl_base_transform


class BYOLTransform():
    """ This the contrastive data augmentation used by the BYOL method. """

    def __init__(self, image_size, normalize):
        self.transform1 = get_ssl_base_transform(image_size,
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

        self.transform2 = get_ssl_base_transform(image_size,
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

    def __call__(self, x):
        """ Perform data augmentation. """

        x1 = self.transform1(x)
        x2 = self.transform2(x)
        return x1, x2
