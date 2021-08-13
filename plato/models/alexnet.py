"""
The AlexNet model for PyTorch.

Reference:

Krizhevsky, Alex. "One weird trick for parallelizing convolutional neural networks."
arXiv preprint arXiv:1404.5997. 2014.

"""

import torchvision
from plato.config import Config


class Model():
    """The AlexNet model."""
    @staticmethod
    def get_model(*args):
        """Obtaining an instance of the AlexNet model."""

        # If True, will return a AlexNet model pre-trained on ImageNet
        pretrained = Config().trainer.pretrained if hasattr(
            Config().trainer, 'pretrained') else False

        return torchvision.models.alexnet(pretrained=pretrained)
