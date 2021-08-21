"""
The ShuffleNet V2 model for PyTorch.

Reference:

Ma, Ningning, et al. "Shufflenet v2: Practical guidelines for efficient cnn architecture design."
Proceedings of the European conference on computer vision (ECCV). 2018.

"""

import torchvision
from plato.config import Config


class Model():
    """The ShuffleNet V2 model."""
    @staticmethod
    def is_valid_model_type(model_type):
        return (model_type.startswith('shufflenet')
                and len(model_type.split('_')) == 2
                and model_type.split('_')[1].isfloat()
                and float(model_type.split('_')[1]) in [0.5, 1.0, 1.5, 2.0])

    @staticmethod
    def get_model(model_type):
        """Obtaining an instance of the ShuffleNet V2 model."""

        # If True, will return a ShuffleNet V2 model pre-trained on ImageNet
        pretrained = Config().trainer.pretrained if hasattr(
            Config().trainer, 'pretrained') else False

        if model_type == 'shufflenet_0.5':
            return torchvision.models.shufflenet_v2_x0_5(pretrained=pretrained)
        elif model_type == 'shufflenet_1.0':
            return torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
        elif model_type == 'shufflenet_1.5':
            return torchvision.models.shufflenet_v2_x1_5(pretrained=pretrained)
        elif model_type == 'shufflenet_2.0':
            return torchvision.models.shufflenet_v2_x2_0(pretrained=pretrained)
