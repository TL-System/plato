"""
Implement the model, including the encoder and the projection, for the SimSiam method.

As we strictly follows the structure and names in SimSiam's paper, therefore,
the encoder of SimSiam contains the backbone and a projector.

The backbone is responsible to obtain the encoded feature of the input data.

"""

from torch import nn

from plato.models import encoders_register
from plato.models import general_mlps_register


class ProjectionMLP(nn.Module):
    """ The implementation of SimCLR's projection layer. """

    def __init__(self, in_dim):
        super().__init__()

        self.layers = general_mlps_register.Model.get_model(
            model_type="simsiam_projection_mlp", input_dim=in_dim)

    def forward(self, x):
        """ Forward the projection layer. """
        for layer in self.layers:
            x = layer(x)

        return x

    def output_dim(self):
        """ Obtain the output dimension. """
        return self.layers[-1].fc.out_features


class PredictionMLP(nn.Module):
    """ The implementation of SimSiam's prediction layer. """

    def __init__(self, in_dim):
        super().__init__()

        self.layers = general_mlps_register.Model.get_model(
            model_type="simsiam_prediction_mlp", input_dim=in_dim)

    def forward(self, x):
        """ Forward the projection layer. """
        for layer in self.layers:
            x = layer(x)

        return x

    def output_dim(self):
        """ Obtain the output dimension. """
        return self.layers[-1][-1].out_features


class BackbonewithProjection(nn.Module):
    """ The module combining the backbone and the projection. """

    def __init__(self, backbone=None, backbone_dim=None):
        super().__init__()

        # define the backbone based on the model_name in config
        if backbone is None:
            self.backbone, self.backbone_dim = encoders_register.get()
        # utilize the custom model
        else:
            self.backbone, self.backbone_dim = backbone, backbone_dim

        # build the projector proposed in the bylo net
        self.projector = ProjectionMLP(in_dim=self.backbone_dim)

        self.projection_dim = self.projector.output_dim()

    def forward(self, x):
        """ Forward the encoder and the projection """
        x = self.backbone(x)
        x = self.projector(x)
        return x

    def get_projection_dim(self):
        return self.projection_dim


class SimSiam(nn.Module):
    """ The implementation of SimSiam method. """

    def __init__(self, backbone=None, backbone_dim=None):
        super().__init__()

        # define the encoder mentioned in the original paper
        # we need to figure out that:
        # in the original paper, the combination of encoder
        # and the projector is regarded as the encoder
        # However, in our implementation, we need to set it
        # to the name 'compound_encoder' to show that it contains
        # the backbone and the projector. Thus, the actual encoder
        # used by the subsequence is the backbone within this
        # compound_encoder.
        # In summary, we utilize different name with the initial paper
        # just to service our Plato's implementation of SSL.
        self.compound_encoder = BackbonewithProjection(backbone, backbone_dim)

        # build the prediciton proposed in the SimSiam net
        self.predictor = PredictionMLP(
            in_dim=self.compound_encoder.get_projection_dim())

    def forward(self, augmented_samples):
        """ Forward two batch of contrastive samples. """
        samples1, samples2 = augmented_samples
        encoded_h1 = self.compound_encoder(samples1)
        encoded_h2 = self.compound_encoder(samples2)

        predicted_z1 = self.predictor(encoded_h1)
        predicted_z2 = self.predictor(encoded_h2)
        return (encoded_h1, encoded_h2), (predicted_z1, predicted_z2)

    @property
    def encoder(self):
        """ Obtain the SimSiam's encoder, i.e., the backbone of the
            defined compound_encoder. """
        return self.compound_encoder.backbone

    @property
    def encode_dim(self):
        """ Obtain the backbone's encoder. """
        return self.compound_encoder.backbone_dim