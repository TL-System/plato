"""
Evaluation of text-to-image generation by showing the DIF scores 
beteen the generated images and the real images.
"""
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image

import stable_diffusion_gen


def generate_images(
    model_path: str,
    generate_config: dict,
    base_context: str = "a photo of a {}",
):
    """Generating images of a trained pseudo-word."""
    # set 4 here because of the limitation in GPU/MPS memory
    n_upper_bound = 6
    n_prompt_images = generate_config["n_prompt_images"]

    learned_token = torch.load(model_path)
    placeholder_token = next(iter(learned_token))

    prompts = stable_diffusion_gen.prepare_prompts(base_context, placeholder_token)
    n_total_images = len(prompts) * n_prompt_images

    n_iters = n_total_images // n_upper_bound
    generate_config["n_prompt_images"] = n_upper_bound

    for _ in range(n_iters):
        stable_diffusion_gen.t2i_generation(
            prompts,
            learned_token,
            placeholder_token,
            generation_config=generate_config,
            is_save_pretrained=False,
        )


def preprocess_image(image):
    """Convert the image to pytorch required."""
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    # return F.center_crop(image, (256, 256))
    return F.interpolate(image, size=(256, 256), mode="bicubic", align_corners=False)


def load_images_to_fid(images_path):
    """Loading the images to be the required type of fid."""

    # Use the glob function to list all image files in the folder
    image_files_path = (
        glob.glob(images_path + "/*.jpg")
        + glob.glob(images_path + "/*.jpeg")
        + glob.glob(images_path + "/*.png")
    )

    loaded_images = [
        np.array(Image.open(path).convert("RGB")) for path in image_files_path
    ]
    n_loaded_images = len(loaded_images)
    print(f"Loaded #{n_loaded_images} images from {images_path}")

    # [n_image, 3, h, w] -
    return torch.cat([preprocess_image(image) for image in loaded_images])


def align_image_sets(real_images, fake_images, is_padding=True):
    """Making two sets of images be the same number."""

    n_real_images = real_images.shape[0]
    n_fake_images = fake_images.shape[0]

    print(f"Having #{n_real_images} real images.")
    print(f"Having #{n_fake_images} fake images.")

    if n_fake_images > n_real_images:
        if is_padding:
            real_images = real_images.repeat(n_fake_images, 1, 1, 1)
            print(f"padding real images to #{n_fake_images}.")
        else:
            selected_indices = torch.randperm(n_fake_images)[:n_real_images]
            fake_images = fake_images[selected_indices]
            print(f"sampling fake images to #{fake_images.shape[0]}.")
    else:
        if is_padding:
            fake_images = fake_images.repeat(n_real_images, 1, 1, 1)
            print(f"padding fake images to #{n_real_images}.")
        else:
            selected_indices = torch.randperm(n_real_images)[:n_fake_images]
            real_images = real_images[selected_indices]
            print(f"sampling real images to #{real_images.shape[0]}.")

    return real_images, fake_images


def compute_two_sets_fid(images_x1: Tensor, images_x2: Tensor):
    """Computing the fid between two sets of images.

    :param images_x1 or images_x2: A `Tensor` holding the images in tensor
     with shape, [n_images, 3, h, w]

    """

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(images_x1, real=True)
    fid.update(images_x2, real=False)

    return {"FID": float(fid.compute())}
