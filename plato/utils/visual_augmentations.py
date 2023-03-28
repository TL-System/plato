"""
The implementation of basic visual augmentations.

Most visual normalizations strictly follows the 'dataset_normalizations' of
https://github.com/Lightning-AI/lightning-bolts (lightning-bolts).

Sources of normalizations:
    - MNIST
        mean=(0.1307, ), std=(0.3081, ) from Plato's source code.
        mean=(0.173, ), std=(0.332, ) from the lightning-bolts.

    - CIFAR10
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])] from
            * https://github.com/leftthomas/SimCLR/blob/master/main.py
        mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262] from
            * https://github.com/mpatacchiola/self-supervised-relational-reasoning
            * lightning-bolts
        mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616))
            * https://www.inside-machinelearning.com/en/why-and-how-to-normalize-data-object-detection-on-image-in-pytorch-part-1/

    - CIFAR100
        mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
            https://github.com/mpatacchiola/self-supervised-relational-reasoning

    - IMAGENET:
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] from
            * https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py
            * https://github.com/PatrickHua/SimSiam/blob/main/main.py
            * lightning-bolts

    - STL10:
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            * https://github.com/mpatacchiola/self-supervised-relational-reasoning
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            * https://github.com/DarkFaceMonster/Pytorch-STL10/blob/master/model.ipynb

        mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27)
            * lightning-bolts's /models/self_supervised/cpc/transforms.py

"""

from typing import Tuple, List, Union

from torchvision import transforms
from PIL import Image, ImageOps


datasets_norm = {
    "MNIST": [
        [
            0.1307,
        ],
        [
            0.3081,
        ],
    ],
    "FashionMNIST": [
        [
            0.1307,
        ],
        [
            0.3081,
        ],
    ],
    "CIFAR10": [[0.491, 0.482, 0.447], [0.247, 0.243, 0.262]],
    "CIFAR100": [[0.491, 0.482, 0.447], [0.247, 0.243, 0.262]],
    "IMAGENET": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "STL10": [[0.4914, 0.4823, 0.4466], [0.247, 0.243, 0.261]],
}


class Solarization:
    """Behave as the Image Filter"""

    def __init__(self, threshold: int = 128):
        self.threshold = threshold

    def __call__(self, image):
        return ImageOps.solarize(image, self.threshold)


class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)


def get_visual_transform(
    image_size: Union[int, Tuple[int, int]],
    normalize: List[list, list],
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    blur_kernel_size: float = 0.0,
    blur_sigma: tuple = (0.1, 2.0),
    color_jitter_prob: float = 0.8,
    gray_scale_prob: float = 0.2,
    horizontal_flip_prob: float = 0.5,
    gaussian_prob: float = 0.5,
    solarization_prob: float = 0.0,
    equalization_prob: float = 0.0,
    min_scale: float = 0.08,
    max_scale: float = 1.0,
    crop_size: int = 32,
):
    """Get the target transformation.

    :param image_size: A tuple or integer containing the input image size.
    :param normalize: A neste list containing the mean and std of the normalization.
    :param brightness: Sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
    :param contrast: sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
    :param saturation: sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
    :param hue: sampled uniformly in [-hue, hue].
    :param blur_kernel_size: kernel size of [h, w] for GaussianBlur.
    :param blur_sigma: sampled uniformly in [-hue, hue].
    :param color_jitter_prob: The probability of applying color jitter.
            Defaults to 0.8.
    :param gray_scale_prob: The probability of converting to gray scale.
            Defaults to 0.2.
    :param horizontal_flip_prob: The probability of flipping horizontally.
            Defaults to 0.5.
    :param gaussian_prob: The probability of applying gaussian blur.
            Defaults to 0.5.
    :param solarization_prob: The probability of applying solarization.
            Defaults to 0.0.
    :param equalization_prob: The probability of applying equalization.
            Defaults to 0.0.
    :param min_scale: Minimum scale of the crops. Defaults to 0.08.
    :param max_scale: Maximum scale of the crops. Defaults to 1.0.
    :param crop_size: Size of the crop. Defaults to 32.
    """
    if blur_kernel_size == 0.0:
        blur_kernel_size = image_size // 20 * 2 + 1

    transform_funcs = [
        transforms.RandomResizedCrop(
            (crop_size, crop_size),
            scale=(min_scale, max_scale),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness, contrast, saturation, hue)],
            p=color_jitter_prob,
        ),
        transforms.RandomGrayscale(p=gray_scale_prob),
        transforms.RandomApply(
            [transforms.GaussianBlur(blur_kernel_size, sigma=blur_sigma)],
            p=gaussian_prob,
        ),
        transforms.RandomApply([Solarization()], p=solarization_prob),
        transforms.RandomApply([Equalization()], p=equalization_prob),
        transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
        transforms.ToTensor(),
    ]
    if normalize is not None:
        transform_funcs.append(transforms.Normalize(*normalize))

    return transforms.Compose(transform_funcs), transform_funcs


class BYOLTransform:
    """This the contrastive data augmentation [1][2] used by the BYOL [3] method.

    [1]. https://github.com/lucidrains/byol-pytorch
    [2]. https://github.com/PatrickHua/SimSiam
    [3]. Grill, et.al, Bootstrap your own latent: A new approach
        to self-supervised Learning, 2021.
    """

    def __init__(
        self, image_size: Union[int, Tuple[int, int]], normalize: List[List[float]]
    ):
        self.transform1, transform_funcs1 = get_visual_transform(
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
            crop_size=image_size,
        )

        self.transform2, transform_funcs2 = get_visual_transform(
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
            crop_size=image_size,
        )

        self.transform_funcs = [transform_funcs1, transform_funcs2]

    def __call__(self, x):
        """Perform data augmentation."""

        x1 = self.transform1(x)
        x2 = self.transform2(x)
        return x1, x2


class MoCoTransform:
    """This the contrastive data augmentation [1] used by the MoCo [2] method.

    [1]. https://github.com/facebookresearch/moco
    [3]. He, et.al, Momentum Contrast for Unsupervised Visual Representation
        Learning, 2020.
    """

    def __init__(
        self, image_size: Union[int, Tuple[int, int]], normalize: List[List[float]]
    ):
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
            crop_size=image_size,
        )
        self.transform_funcs = transform_funcs

    def __call__(self, x):
        """Perform the contrastive data augmentation."""
        x1 = self.transform(x)
        x2 = self.transform(x)

        return x1, x2


class SimCLRTransform:
    """This the contrastive data augmentation [1][2] used by the SimCLR [3] method.

    [1]. https://github.com/google-research/simclr
    [2]. https://github.com/PatrickHua/SimSiam
    [3]. Chen, et.al, A Simple Framework for Contrastive Learning of
        Visual Representations, 2020.
    """

    def __init__(
        self, image_size: Union[int, Tuple[int, int]], normalize: List[List[float]]
    ):
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
            crop_size=image_size,
        )
        self.transform_funcs = transform_funcs

    def __call__(self, x):
        """Perform the contrastive data augmentation."""
        x1 = self.transform(x)
        x2 = self.transform(x)

        return x1, x2


class SimSiamTransform:
    """This the contrastive data augmentation [1] used by the Simsiam [2] method.

    [1]. https://github.com/PatrickHua/SimSiam
    [2]. Chen & He, Exploring Simple Siamese Representation Learning, 2021.
    """

    def __init__(
        self, image_size: Union[int, Tuple[int, int]], normalize: List[List[float]]
    ):
        # by default simsiam use image size 224
        image_size = 224 if image_size is None else image_size
        p_blur = 0.5 if image_size > 32 else 0  # exclude cifar
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
            gaussian_prob=p_blur,
            solarization_prob=0.0,
            equalization_prob=0.0,
            min_scale=0.2,
            max_scale=1.0,
            crop_size=image_size,
        )
        self.transform_funcs = transform_funcs

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


class SvAVTransform:
    """This the contrastive data augmentation [1] used by the SWAV [2] method.

    [1]. https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
    [2]. Caron, et.al, Unsupervised Learning of Visual Features by Contrasting
        Cluster Assignments, 2020.
    """

    def __init__(
        self, image_size: Union[int, Tuple[int, int]], normalize: List[List[float]]
    ):
        p_blur = 0.5 if image_size > 32 else 0  # exclude cifar
        self.transform, transform_funcs = get_visual_transform(
            image_size,
            normalize,
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
            color_jitter_prob=0.8,
            gray_scale_prob=0.2,
            horizontal_flip_prob=0.5,
            gaussian_prob=p_blur,
            solarization_prob=0.0,
            equalization_prob=0.0,
            min_scale=0.08,
            max_scale=1.0,
            crop_size=image_size,
        )
        self.transform_funcs = transform_funcs

    def __call__(self, x):
        return self.transform(x)
