"""
The augmentations used for constrative learning.


"""

import logging

from plato.config import Config

from .simsiam_aug import SimSiamTransform
from .byol_aug import BYOLTransform
from .simclr_aug import SimCLRTransform
from .test_aug import TestTransform

from .normalizations import datasets_norm


def get(name='simsiam', train=True, for_downstream_task=False):
    """ Get the data agumentation for different methods, and the final
        linear evaluation part. """
    transform_mapper = {
        "simsiam": SimSiamTransform,
        "byol": BYOLTransform,
        "simclr": SimCLRTransform,
        "test": TestTransform,
    }
    supported_transform_name = list(transform_mapper.keys())
    if name not in supported_transform_name:
        logging.exception(("%s is not included in the support set %s"), name,
                          supported_transform_name)
        raise NotImplementedError

    dataset_name = Config().data.datasource
    is_norm = Config().data.is_norm
    image_size = Config().trainer.image_size

    normalize = datasets_norm[dataset_name] if is_norm else None

    # obtain the augmentation transform  for the ssl train
    # train: True, for_downstream_task: False
    if train and not for_downstream_task:
        augmentation = transform_mapper[name](image_size, normalize)
    # obtain the transform for the train stage of the for_downstream_task
    # train: True, for_downstream_task: True
    elif train and for_downstream_task:
        augmentation = transform_mapper["test"](image_size,
                                                train=True,
                                                normalize=normalize)
    # obtain the transform for the test stage of the downstream_task
    # train: False, for_downstream_task: True
    elif not train and for_downstream_task:
        augmentation = transform_mapper["test"](image_size,
                                                train=False,
                                                normalize=normalize)
    # obtain the transform for the monitor stage of the ssl
    # train: False, for_downstream_task: False
    else:
        augmentation = transform_mapper["test"](image_size,
                                                train=False,
                                                normalize=normalize)
    return augmentation
