"""
The implementation of logging model'size and model structure to the file.

"""

import os

import torch
from torchinfo import summary

from plato.config import Config

data_image_info = {
    "CIFAR10": (3, 32, 32),
    "MNIST": (1, 28, 28),
    "STL10": (3, 96, 96)
}


def log_model_information(model,
                          image_source=None,
                          image_size=None,
                          file_name=None,
                          location=None):
    """ Save the model information, including structure and size to the logging file.

        The image_size should be a 3d tuple, (channel, h, w)
    """

    if image_size is None:
        assert image_source is not None
        image_size = data_image_info[image_source]
    location = "./" if location is None else location
    file_name = "model_info.log" if file_name is None else file_name

    to_save_path = os.path.join(location, file_name)

    obtained_device = Config().device()

    model_stats = summary(model, image_size, device=obtained_device)
    summary_str = str(model_stats)
    with open(to_save_path, 'w') as f:
        f.write(summary_str)