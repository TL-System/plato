"""
The SqueezeNet model and SqueezeNet 1.1 model for PyTorch.

SqueezeNet model architecture is from the
“SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size” paper.

SqueezeNet 1.1 model is from the official SqueezeNet repo.
https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1
SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0,
without sacrificing accuracy.
"""

import torchvision
from plato.config import Config


class Model():
    """The SqueezeNet model."""
    @staticmethod
    def is_valid_model_type(model_type):
        return (model_type == 'squeezenet_0') or (model_type == 'squeezenet_1')

    @staticmethod
    def get_model(model_type):
        """Obtain an instance of this SqueezeNet model."""
        if not Model.is_valid_model_type(model_type):
            raise ValueError(
                'Invalid SqueezeNet model type: {}'.format(model_type))

        # If True, will return a SqueezeNet model pre-trained on ImageNet
        pretrained = Config().trainer.pretrained if hasattr(
            Config().trainer, 'pretrained') else False

        if model_type == 'squeezenet_0':
            return torchvision.models.squeezenet1_0(pretrained=pretrained)
        elif model_type == 'squeezenet_1':
            return torchvision.models.squeezenet1_1(pretrained=pretrained)
