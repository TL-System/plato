"""
Implementation of the encoder register

    For the encoder with ResNet as backbone, we do not utilize
the ones implemented under the plato/models. The main reason is
that the plato's version ignore the AdaptiveAvgPool2d. Then, all current
methods of self-supervised utilize the resnet from torchvision to perform
the learning process. For the fair comparsion, we decide to utilize the
one from torchvision directly.

Generally, for the self-supervised learning, the encoder is defined as the
general network, such as VGG and ResNet, but ignoring the final fully-connected
blocks.

"""

import logging

from torch import nn
import torchvision

from plato.models import registry as model_register
from plato.config import Config


class TruncatedLeNetModel(nn.Module):
    """The truncated LeNet-5 model.

    """

    def __init__(self, defined_lenet5_model):
        super().__init__()
        self.model = defined_lenet5_model
        self.model.fc4 = nn.Identity()
        self.model.relu4 = nn.Identity()
        self.model.fc5 = nn.Identity()

    def forward(self, samples):
        """ Forward to specific layer (cut)_layer) of LeNet5. """
        return self.model.forward_to(samples, cut_layer="flatten")


def get():
    """ Register the encoder from the required model by removing the
        final fully-connected blocks. """

    model_name = Config().trainer.model_name
    datasource = Config().data.datasource
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
        encoder.fc = nn.Identity()

    if "resnet" in model_name:
        resnets = {
            "resnet_18": torchvision.models.resnet18,
            "resnet_50": torchvision.models.resnet50,
        }

        num_classes = 10
        if hasattr(Config().data, 'num_classes'):
            num_classes = Config().data.num_classes
        encoder = resnets[model_name](num_classes=num_classes)

        if datasource == "CIFAR10":
            # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
            # See Section B.9 of SimCLR paper.
            encoder.conv1 = nn.Conv2d(3,
                                      64,
                                      kernel_size=3,
                                      stride=1,
                                      padding=2,
                                      bias=False)
            encoder.maxpool = nn.Identity()

        # get encoding dimensions
        #   i.e., the output dim of the encoder
        encode_output_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

    return encoder, encode_output_dim
