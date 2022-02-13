"""
The RegNet model.

Ilija Radosavovic, et al., "Designing Network Design Spaces,"
https://arxiv.org/abs/2003.13678
"""

import torchvision
from plato.config import Config


class Model():
    """The RegNet model."""
    @staticmethod
    def get_model(model_type):
        """Obtaining an instance of the RegNet model."""

        # If True, will return a RegNet model pre-trained on ImageNet
        pretrained = Config().trainer.pretrained if hasattr(
            Config().trainer, 'pretrained') else False

        if model_type == 'regnet_x_400mf':
            return torchvision.models.regnet_x_400mf(pretrained=pretrained)
        if model_type == 'regnet_x_800mf':
            return torchvision.models.regnet_x_800mf(pretrained=pretrained)
        if model_type == 'regnet_x_1_6gf':
            return torchvision.models.regnet_x_1_6gf(pretrained=pretrained)
        if model_type == 'regnet_x_3_2gf':
            return torchvision.models.regnet_x_3_2gf(pretrained=pretrained)
        if model_type == 'regnet_x_8gf':
            return torchvision.models.regnet_x_8gf(pretrained=pretrained)
        if model_type == 'regnet_x_16gf':
            return torchvision.models.regnet_x_16gf(pretrained=pretrained)
        if model_type == 'regnet_x_32gf':
            return torchvision.models.regnet_x_32gf(pretrained=pretrained)
        if model_type == 'regnet_y_400mf':
            return torchvision.models.regnet_y_400mf(pretrained=pretrained)
        if model_type == 'regnet_y_800mf':
            return torchvision.models.regnet_y_800mf(pretrained=pretrained)
        if model_type == 'regnet_y_1_6gf':
            return torchvision.models.regnet_y_1_6gf(pretrained=pretrained)
        if model_type == 'regnet_y_3_2gf':
            return torchvision.models.regnet_y_3_2gf(pretrained=pretrained)
        if model_type == 'regnet_y_8gf':
            return torchvision.models.regnet_y_8gf(pretrained=pretrained)
        if model_type == 'regnet_y_16gf':
            return torchvision.models.regnet_y_16gf(pretrained=pretrained)
        if model_type == 'regnet_y_32gf':
            return torchvision.models.regnet_y_32gf(pretrained=pretrained)
