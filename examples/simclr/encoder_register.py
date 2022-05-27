"""
Implementation of the encoder register

"""

import logging

import torch.nn as nn
from torchvision.models import resnet18, resnet50, vgg16

from lenet5 import Model as lenet5_model


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def register_encoder(base_model_name):
    """ Register the encoder from the required model by removing the 
        final fully-connected blocks. """

    logging.info((
        "Define the encoder from the model: {} without final fully-connected layers"
    ).format(base_model_name))
    if base_model_name == "lenet5":
        """ Using my own lenet5_model. """
        encoder = lenet5_model()
        # get dimensions of classifier layer
        #   i.e., the output dim of the encoder
        encode_output_dim = encoder.fc[0][0].in_features
        encoder.fc = Identity()

    if base_model_name == "vgg_16":
        encoder = vgg16(pretrained=False)
        # get dimensions of classifier layer
        #   i.e., the output dim of the encoder
        encode_output_dim = encoder.classifier.in_features
        encoder.classifier = Identity()

    if base_model_name == "resnet_18":
        encoder = resnet18(pretrained=False)
        # get dimensions of classifier layer
        #   i.e., the output dim of the encoder
        encode_output_dim = encoder.fc.in_features
        encoder.fc = Identity()

    if base_model_name == "resnet_50":
        encoder = resnet50(pretrained=False)
        # get dimensions of classifier layer
        #   i.e., the output dim of the encoder
        encode_output_dim = encoder.fc.in_features
        encoder.fc = Identity()

    return encoder, encode_output_dim