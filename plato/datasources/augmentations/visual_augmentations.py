"""
The implementation of basic visual augmentations.

"""
from typing import Tuple, List

from torchvision import transforms
from PIL import Image, ImageOps


class Solarization:
    """Behave as the Image Filter"""

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, image):
        return ImageOps.solarize(image, self.threshold)


class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)


def get_visual_transform(
    image_size: Tuple[int, int],
    normalize: List[list, list],
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    kernel_size: float = 0.0,
    blur_sigma: list = [0.1, 2.0],
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

    :param image_size: A tuple containing the input image size.
    :param normalize: A neste list containing the mean and std of the normalization.
    :param brightness: Sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
    :param contrast: sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
    :param saturation: sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
    :param hue: sampled uniformly in [-hue, hue].
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
    if kernel_size == 0.0:
        kernel_size = image_size // 20 * 2 + 1

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
            [transforms.GaussianBlur(kernel_size, sigma=blur_sigma)], p=gaussian_prob
        ),
        transforms.RandomApply([Solarization()], p=solarization_prob),
        transforms.RandomApply([Equalization()], p=equalization_prob),
        transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
        transforms.ToTensor(),
    ]
    if normalize is not None:
        transform_funcs.append(transforms.Normalize(*normalize))

    return transforms.Compose(transform_funcs), transform_funcs
