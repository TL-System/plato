"""
The implementation of the model structure used in the FedRep method,
which is the combination of
    - backbone encoder
    - multiple fc layer.

The backbone encoder is used the global model while fc layer
is used as the personalized model.

In the authors' original paper, only the model's final fc that generates the
classification outputs is used as the personalized model. For example, in the
vgg model, the fedrep's model structure is
the global model: conv + fc1 + fc2
the personalized model: fc3


However, in our implementation, we remove all fc layers but build one lienar fc
layer to combine with the convolutional layers. The target of this mechanism
is mainly to make a fair comparsion with other methods. This does not present
negative impact on the performance of the fedrep.

Currently, the training model of all methods have the similar complexity.

"""

from torch import nn

from plato.models import encoders_register
from plato.models import general_mlps_register as general_MLP_model


class BackBoneEnc(nn.Module):
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

        return BackBoneEnc()
