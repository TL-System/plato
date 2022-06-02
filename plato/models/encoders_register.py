"""
Implementation of the encoder register

    For the encoder with ResNet as backbone, we do not utilize 
the ones implemented under the plato/models. The main reason is 
that the plato's version ignore the AdaptiveAvgPool2d. Then, all current
methods of self-supervised utilize the resnet from torchvision to perform
the learning process. For the fair comparsion, we decide to utilize the 
one from torchvision directly.


"""

import logging

from torch import nn
import torchvision

from plato.models import registry as model_register
from plato.config import Config


class Identity(nn.Module):
    """ The constant layer. """

    def forward(self, samples):
        """ Forward the input without any operations. """
        return samples


class TruncatedLeNetModel(nn.Module):
    """The truncated LeNet-5 model.

    Arguments:
        num_classes (int): The number of classes. Default: 10.
    """

    def __init__(self, defined_lenet5_model):
        super().__init__()
        self.model = defined_lenet5_model
        self.model.fc4 = Identity()
        self.model.relu4 = Identity()
        self.model.fc5 = Identity()

    def forward(self, samples):
        """ Forward to specific layer (cut)_layer) of LeNet5. """
        return self.model.forward_to(samples, cut_layer="flatten")


def get():
    """ Register the encoder from the required model by removing the
        final fully-connected blocks. """

    model_name = Config().trainer.model_name

    logging.info(
        "Define the encoder from the model: %s without final fully-connected layers",
        model_name)

    if model_name == "lenet5":
        model = model_register.get()
        # get encoding dimensions
        #   i.e., the output dim of the encoder
        encode_output_dim = model.fc4.in_features
        encoder = TruncatedLeNetModel(model)

    if "vgg" in model_name:
        encoder = model_register.get()
        # get encoding dimensions
        #   i.e., the output dim of the encoder
        encode_output_dim = encoder.fc.in_features
        encoder.fc = Identity()

    if "resnet" in model_name:
        resnets = {
            "resnet_18": torchvision.models.resnet18,
            "resnet_50": torchvision.models.resnet50,
        }

        num_classes = 10
        if hasattr(Config().data, 'num_classes'):
            num_classes = Config().data.num_classes
        encoder = resnets[model_name](num_classes=num_classes)

        # get encoding dimensions
        #   i.e., the output dim of the encoder
        encode_output_dim = encoder.fc.in_features
        encoder.fc = Identity()

    return encoder, encode_output_dim
