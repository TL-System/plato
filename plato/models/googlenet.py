"""
The GoogLeNet (Inception v1) model for PyTorch.

Reference:

Szegedy, Christian, et al. "Going deeper with convolutions."
Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

"""

import torchvision
from plato.config import Config


class Model():
    """The GoogleNet model."""
    @staticmethod
    def get_model(*args):
        """Obtaining an instance of the GoogleNet model."""

        # If True, will return a GoogleNet model pre-trained on ImageNet
        pretrained = Config().trainer.pretrained if hasattr(
            Config().trainer, 'pretrained') else False

        # If True, preprocesses the input according to the method
        # with which it was trained on ImageNet
        transform_input = Config().trainer.transform_input if hasattr(
            Config().trainer, 'transform_input') else False

        return torchvision.models.googlenet(pretrained=pretrained,
                                            aux_logits=False,
                                            transform_input=transform_input)
