"""
A factory that generates fully convolutional neural network encoder.

The fully convolutional neural network used as the encoder in this implementation
is the backbone part of a model.

This encoder is solely capable of extracting features from the input sample.
Generally, in the context of classification tasks, this encoder has to cooperate
with the classification head to make the prediction.

Besides, the 'AdaptiveAvgPool2d' layer is included to support extracting
features with fixed dimensions.

"""

from typing import Optional, Dict

from torch import nn
import torchvision

from plato.models.lenet5 import Model as lenet5_model
from plato.models.vgg import Model as vgg_model

from plato.config import Config


class TruncatedLeNetModel(nn.Module):
    """The truncated LeNet-5 model."""

    def __init__(self, defined_lenet5_model):
        super().__init__()
        self.model = defined_lenet5_model
        self.model.fc4 = nn.Identity()
        self.model.relu4 = nn.Identity()
        self.model.fc5 = nn.Identity()

    def forward(self, samples):
        """Forward to specific layer (cut)_layer) of LeNet5."""
        self.model.cut_layer = "flatten"
        return self.model.forward_to(samples)


class Model:
    """The encoder obtained by removing the final
    fully-connected blocks of the required model.
    """

    # pylint:disable=too-few-public-methods
    @staticmethod
    def get(
        model_name: Optional[str] = None, **kwargs: Dict[str, str]
    ):  # pylint: disable=unused-argument
        """Returns an encoder that is a fully CNN block."""

        # as the final fully-connected layer will be removed
        # the number of classes can be the randomly value
        # thus, set it to be constant value 10.
        num_classes = 10

        if model_name == "lenet5":
            model = lenet5_model(num_classes=num_classes)
            # get encoding dimensions
            #   i.e., the output dim of the encoder
            encode_output_dim = model.fc4.in_features
            encoder = TruncatedLeNetModel(model)

        if "vgg" in model_name:
            encoder = vgg_model.get(model_name=model_name, num_classes=num_classes)
            # get encoding dimensions
            #   i.e., the output dim of the encoder
            encode_output_dim = encoder.fc.in_features
            encoder.fc = nn.Identity()

        if "resnet" in model_name:
            resnets = {
                "resnet_18": torchvision.models.resnet18,
                "resnet_50": torchvision.models.resnet50,
            }

            encoder = resnets[model_name](num_classes=num_classes)

            datasource = (
                kwargs["datasource"]
                if "datasource" in kwargs
                else Config().data.datasource
            )
            if "CIFAR" in datasource:
                # The structure specifically for CIFAR-based dataset.
                # Replace conv 7x7 with conv 3x3,
                # and remove first max pooling.
                # For example,
                #   see Section B.9 of SimCLR paper.
                encoder.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                encoder.maxpool = nn.Identity()

            # get encoding dimensions
            #   i.e., the output dim of the encoder
            encode_output_dim = encoder.fc.in_features
            encoder.fc = nn.Identity()

        encoder.encoding_dim = encode_output_dim

        return encoder
