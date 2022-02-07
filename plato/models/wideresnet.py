"""
The WideResNet model.

This model is the same as ResNet except for the bottleneck number of channels which
is twice larger in every block.

S. Zagoruyko, N. Komodakis, "Wide Residual Networks,"
https://arxiv.org/pdf/1605.07146.pdf
"""

import torchvision
from plato.config import Config


class Model():
    """The Wide ResNet model."""

    @staticmethod
    def get_model(model_type):
        """Obtaining an instance of the Wide ResNet model."""

        # If True, will return a Wide ResNet model pre-trained on ImageNet
        pretrained = Config().trainer.pretrained if hasattr(
            Config().trainer, 'pretrained') else False

        if model_type == 'wide_resnet50_2':
            return torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif model_type == 'wide_resnet101_2':
            return torchvision.models.wide_resnet101_2(pretrained=pretrained)
