"""
The EfficientNet model.

Mingxing Tan and Quoc V. Le, 
"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,"
http://proceedings.mlr.press/v97/tan19a/tan19a.pdf
"""

import torchvision
from plato.config import Config


class Model():
    """The EfficientNet model."""
    @staticmethod
    def get_model(model_type):
        """Obtaining an instance of the EfficientNet model."""

        # If True, will return a EfficientNet model pre-trained on ImageNet
        pretrained = Config().trainer.pretrained if hasattr(
            Config().trainer, 'pretrained') else False

        if model_type == 'efficientnet_b0':
            return torchvision.models.efficientnet_b0(pretrained=pretrained)
        if model_type == 'efficientnet_b1':
            return torchvision.models.efficientnet_b1(pretrained=pretrained)
        if model_type == 'efficientnet_b2':
            return torchvision.models.efficientnet_b2(pretrained=pretrained)
        if model_type == 'efficientnet_b3':
            return torchvision.models.efficientnet_b3(pretrained=pretrained)
        if model_type == 'efficientnet_b4':
            return torchvision.models.efficientnet_b4(pretrained=pretrained)
        if model_type == 'efficientnet_b5':
            return torchvision.models.efficientnet_b5(pretrained=pretrained)
        if model_type == 'efficientnet_b6':
            return torchvision.models.efficientnet_b6(pretrained=pretrained)
        elif model_type == 'efficientnet_b7':
            return torchvision.models.efficientnet_b7(pretrained=pretrained)
