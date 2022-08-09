"""The LeNet-5 model for TensorFlow.

Reference:

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to
document recognition." Proceedings of the IEEE, November 1998.
"""
import collections

from tensorflow import keras
from tensorflow.keras import layers

from plato.config import Config


class Model(keras.Model):
    """The LeNet-5 model.

    Arguments:
        num_classes (int): The number of classes.
    """

    def __init__(self, num_classes=10, cut_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.cut_layer = cut_layer

        self.conv1 = layers.Conv2D(
            filters=6, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 1)
        )
        self.pool1 = layers.AveragePooling2D()
        self.conv2 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu")
        self.pool2 = layers.AveragePooling2D()

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(units=120, activation="relu")
        self.fc2 = layers.Dense(units=84, activation="relu")
        self.fc3 = layers.Dense(units=num_classes, activation="softmax")

        # Preparing named layers so that the model can be split and straddle
        # across the client and the server
        self.model_layers = []
        self.layerdict = collections.OrderedDict()
        self.layerdict["conv1"] = self.conv1
        self.layerdict["pool1"] = self.pool1
        self.layerdict["conv2"] = self.conv2
        self.layerdict["pool2"] = self.pool2
        self.layerdict["flatten"] = self.flatten
        self.layerdict["fc1"] = self.fc1
        self.layerdict["fc2"] = self.fc2
        self.layerdict["fc3"] = self.fc3
        self.model_layers.append("conv1")
        self.model_layers.append("pool1")
        self.model_layers.append("conv2")
        self.model_layers.append("pool2")
        self.model_layers.append("flatten")
        self.model_layers.append("fc1")
        self.model_layers.append("fc2")
        self.model_layers.append("fc3")

    def call(self, x):
        """The forward pass."""
        if self.cut_layer is None:
            # If cut_layer is None, use the entire model for training
            # or testing
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        else:
            # Otherwise, use only the layers after the cut_layer
            # for training
            layer_index = self.model_layers.index(self.cut_layer)

            for i in range(layer_index + 1, len(self.model_layers)):
                x = self.layerdict[self.model_layers[i]](x)

        return x

    def call_to(self, x, cut_layer):
        """Extract features using the layers before (and including)
        the cut_layer.
        """
        layer_index = self.model_layers.index(cut_layer)

        for i in range(0, layer_index + 1):
            x = self.layerdict[self.model_layers[i]](x)

        return x

    def build_model(self, input_shape):
        """Building the model using dimensions for the datasource."""
        self.build(input_shape)
        self.summary()
