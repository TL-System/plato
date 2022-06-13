"""
The Inception v3 model for PyTorch.

Reference:

Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision."
Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

"""

import torchvision
from plato.config import Config


class Model():
    """The Inception v3 model."""
    @staticmethod
    def get_model(*args):
        """Obtaining an instance of this model."""

        # If True, will return a Inception v3 model pre-trained on ImageNet
        pretrained = Config().trainer.pretrained if hasattr(
            Config().trainer, 'pretrained') else False

        # If True, preprocesses the input according to the method
        # with which it was trained on ImageNet
        transform_input = Config().trainer.transform_input if hasattr(
            Config().trainer, 'transform_input') else False

        return torchvision.models.inception_v3(pretrained=pretrained,
                                               aux_logits=False,
                                               transform_input=transform_input)
