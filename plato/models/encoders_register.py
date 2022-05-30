"""
Implementation of the encoder register

"""

import logging

import torch.nn as nn

from plato.models import registry as model_register
from plato.config import Config


class Identity(nn.Module):
    """ The constant layer. """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


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

    def forward(self, x):
        return self.model.forward_to(x, cut_layer="flatten")


def get():
    """ Register the encoder from the required model by removing the 
        final fully-connected blocks. """

    model_name = Config().trainer.model_name

    logging.info((
        "Define the encoder from the model: {} without final fully-connected layers"
    ).format(model_name))

    model = model_register.get()

    if model_name == "lenet5":
        """ Using wrapper of lenet5_model. """
        # get encoding dimensions
        #   i.e., the output dim of the encoder
        encode_output_dim = model.fc4.in_features
        encoder = TruncatedLeNetModel(model)

    if "vgg" in model_name:
        encoder = model
        # get encoding dimensions
        #   i.e., the output dim of the encoder
        encode_output_dim = encoder.fc.in_features
        encoder.fc = Identity()

    if "resnet" in model_name:
        encoder = model
        # get encoding dimensions
        #   i.e., the output dim of the encoder
        encode_output_dim = encoder.linear.in_features
        encoder.linear = Identity()

    return encoder, encode_output_dim