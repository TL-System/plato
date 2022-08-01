"""
The implementation of a new resnet model, which is the combination of
backbone encoder and a fc layer.

This is to achieve a fair comparsion with our pFL methods which includes
a encoder and a linear fc layer.

"""

from torch import nn

from plato.models import encoders_register
from plato.models import general_mlps_register as general_MLP_model


class BackBoneCls(nn.Module):
    """ The implementation of a model containing a backbone and a fc layer. """

    def __init__(self):
        super().__init__()

        # define the encoder based on the model_name in config
        self.encoder, self.encode_dim = encoders_register.get()

        self.clf_fc = general_MLP_model.Model.get_model(
            model_type="pure_one_layer_mlp", input_dim=self.encode_dim)

    def forward(self, x):
        """ Forward two batch of contrastive samples. """
        encoded_fea = self.encoder(x)

        out = self.clf_fc(encoded_fea)
        return out

    @staticmethod
    def get_model():

        return BackBoneCls()
