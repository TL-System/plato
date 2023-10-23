"""
Implement the model, including the encoder and the projection, for the contrastive adaptation method.

"""

from torch import nn

from plato.models import encoders_register
from plato.models import general_mlps_register


class ProjectionMLP(nn.Module):
    """ The implementation of SimCLR's projection layer. """

    def __init__(self, in_dim):
        super().__init__()

        self.mlp_layers = general_mlps_register.Model.get_model(
            model_type="simclr_projection_mlp", input_dim=in_dim)

    def forward(self, x):
        """ Forward the projection layer. """

        return self.mlp_layers(x)


class pFLCMANet(nn.Module):
    """ The implementation of ContrasAdap method.

    """

    def __init__(self, encoder=None, encoder_dim=None):
        super().__init__()

        # define the encoder based on the model_name in config
        if encoder is None:
            self.encoder, self.encode_dim = encoders_register.get()
        # utilize the custom model
        else:
            self.encoder, self.encode_dim = encoder, encoder_dim

        # build the projector proposed in the paper
        # this is the general used one, which can be also
        # regarded as the predictor in the paper.
        # For simplicity, we define projector here
        self.projector = ProjectionMLP(in_dim=self.encode_dim)

    def forward(self, augmented_samples):
        """ Forward two batch of contrastive samples. """
        samples1, samples2 = augmented_samples
        encoded_h1 = self.encoder(samples1)
        encoded_h2 = self.encoder(samples2)

        projected_z1 = self.projector(encoded_h1)
        projected_z2 = self.projector(encoded_h2)
        return projected_z1, projected_z2

    @staticmethod
    def get_model():
        """Obtaining an instance of this model provided that the name is valid."""

        return pFLCMANet()