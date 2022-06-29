"""
The implementation of a base transform generation function for the
self-supervised learning methods.

"""

from torchvision import transforms
from PIL import Image, ImageOps


class Solarization():
    """ Behave as the Image Filter """

    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, image):
        return ImageOps.solarize(image, self.threshold)


class Equalization:

    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)


def get_ssl_base_transform(
    image_size,
    normalize,
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
    """Class that applies Cifar10/Cifar100 transformations.

    Args:
        cifar (str): type of cifar, either cifar10 or cifar100.
        brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
        contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
        saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
        hue (float): sampled uniformly in [-hue, hue].
        color_jitter_prob (float, optional): probability of applying color jitter.
            Defaults to 0.8.
        gray_scale_prob (float, optional): probability of converting to gray scale.
            Defaults to 0.2.
        horizontal_flip_prob (float, optional): probability of flipping horizontally.
            Defaults to 0.5.
        gaussian_prob (float, optional): probability of applying gaussian blur.
            Defaults to 0.0.
        solarization_prob (float, optional): probability of applying solarization.
            Defaults to 0.0.
        equalization_prob (float, optional): probability of applying equalization.
            Defaults to 0.0.
        min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
        max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
        crop_size (int, optional): size of the crop. Defaults to 32.
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
            [transforms.GaussianBlur(kernel_size, sigma=blur_sigma)],
            p=gaussian_prob),
        transforms.RandomApply([Solarization()], p=solarization_prob),
        transforms.RandomApply([Equalization()], p=equalization_prob),
        transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
        transforms.ToTensor(),
    ]
    if normalize is not None:
        transform_funcs.append(transforms.Normalize(*normalize))

    return transforms.Compose(transform_funcs)